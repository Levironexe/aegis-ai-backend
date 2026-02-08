"""
LangGraph Agent for Cybersecurity Investigation

This module implements a multi-step reasoning agent using LangGraph for
cybersecurity analysis, threat detection, and incident investigation.

The agent follows a 5-node graph structure:
1. Planning: Assess query and determine investigation approach
2. Tool Selection: Choose appropriate tools based on the plan
3. Tool Execution: Run selected tools to gather information
4. Analysis: Correlate findings and assess threats
5. Response Generation: Format final report for user

The agent maintains compatibility with the existing gateway interface,
returning OpenAI-compatible SSE chunks for seamless frontend integration.
"""

import logging
from typing import AsyncGenerator, List, Dict, Any, Literal, TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from app.config import settings

logger = logging.getLogger(__name__)


# ============ STATE DEFINITION ============

class CyberSecurityState(TypedDict):
    """
    State for the cybersecurity investigation agent.

    This state is passed through all nodes in the graph and maintains
    context throughout the investigation workflow.
    """
    # Core conversation
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Investigation context
    investigation_steps: list[str]  # Sequence of investigation actions taken
    iocs_found: list[dict]  # List of Indicators of Compromise detected
    severity_level: str  # Risk level: "low", "medium", "high", "critical"
    mitre_tactics: list[str]  # MITRE ATT&CK tactics identified

    # Tool execution tracking
    tools_used: list[str]  # Names of tools invoked
    tool_results: list[dict]  # Results from tool executions
    pending_approval: dict | None  # Tool awaiting human approval (future feature)

    # Response generation
    final_response: str  # Generated investigation report


# ============ LANGGRAPH AGENT CLASS ============

class LangGraphAgent:
    """
    LangGraph-powered cybersecurity investigation agent.

    This agent orchestrates multi-step investigations using tools and LLM reasoning.
    It implements the same interface as ClaudeClient/GeminiClient for seamless
    integration with the existing gateway pattern.

    Usage:
        agent = LangGraphAgent()
        agent.register_tools([tool1, tool2, ...])

        async for chunk in agent.stream_chat_completion(
            model="agent/cyber-analyst",
            messages=[{"role": "user", "content": "Analyze this IP..."}],
            temperature=0.7
        ):
            # chunk is in OpenAI-compatible format
            print(chunk)
    """

    def __init__(self):
        """Initialize the agent with LLM and build the investigation graph."""
        # Get model name from settings and map to Anthropic API format
        model_name = getattr(settings, 'agent_model', 'claude-haiku-4-5')

        # Map friendly names to actual Anthropic API model names
        model_mapping = {
            "claude-haiku-4.5": "claude-3-5-haiku-20241022",
            "claude-haiku-4-5": "claude-3-5-haiku-20241022",
            "claude-sonnet-4.5": "claude-3-5-sonnet-20241022",
            "claude-sonnet-4-5": "claude-3-5-sonnet-20241022",
            "claude-opus-4.5": "claude-opus-4-20250514",
            "claude-opus-4-5": "claude-opus-4-20250514",
        }
        anthropic_model = model_mapping.get(model_name, model_name)

        # Initialize LLM for reasoning (using Claude for multi-step reasoning)
        self.llm = ChatAnthropic(
            model=anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0.7,
            max_tokens=4096,
        )

        # Tools will be registered here
        self.tools = []
        self.tool_node = None

        # Build the investigation graph
        self.app = self._build_graph()

        logger.info(f"LangGraph agent initialized with model: {anthropic_model}")

    def register_tools(self, tools: list):
        """
        Register tools for the agent to use during investigations.

        Args:
            tools: List of LangChain tools (use BaseTool.to_langchain_tool())
        """
        self.tools = tools
        if tools:
            self.tool_node = ToolNode(tools)
            # Rebuild graph with tools
            self.app = self._build_graph()
            logger.info(f"Registered {len(tools)} tools: {[t.name for t in tools]}")
        else:
            logger.warning("No tools registered - agent will run in reasoning-only mode")

    def _build_graph(self) -> StateGraph:
        """
        Build the cybersecurity investigation graph.

        Graph structure:

            START
              ↓
          Planning (assess query, determine approach)
              ↓
          Tool Selection (LLM decides which tools to use)
              ↓
          ┌─────────────┐
          │ Use Tools?  │
          └─────────────┘
             ↓       ↓
            Yes      No
             ↓       ↓
          Execute   Skip
          Tools      ↓
             ↓       ↓
          ┌─────────────┐
          │ Continue?   │
          └─────────────┘
             ↓       ↓
            Yes      No
             ↓       ↓
          (loop)  Analysis (correlate findings)
                     ↓
                  Response (generate report)
                     ↓
                    END

        Returns:
            Compiled StateGraph ready for execution
        """
        workflow = StateGraph(CyberSecurityState)

        # Add all nodes to the graph
        workflow.add_node("classify", self._classify_node)  # NEW: Determine if security query
        workflow.add_node("simple_response", self._simple_response_node)  # NEW: For non-security queries
        workflow.add_node("planning", self._planning_node)
        workflow.add_node("tool_selection", self._tool_selection_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("analysis", self._analysis_node)
        workflow.add_node("response", self._response_node)

        # Define entry point
        workflow.set_entry_point("classify")

        # Classify → Security investigation OR Simple response
        workflow.add_conditional_edges(
            "classify",
            self._is_security_query,
            {
                "security": "planning",
                "general": "simple_response"
            }
        )

        # Simple response → END
        workflow.add_edge("simple_response", END)

        # Define edges between security investigation nodes
        workflow.add_edge("planning", "tool_selection")

        # Conditional: use tools or skip to analysis?
        workflow.add_conditional_edges(
            "tool_selection",
            self._should_use_tools,
            {
                "execute": "execute_tools",
                "skip": "analysis"
            }
        )

        # Conditional: continue investigation or move to analysis?
        workflow.add_conditional_edges(
            "execute_tools",
            self._continue_investigation,
            {
                "continue": "tool_selection",  # Loop for multi-step investigation
                "analyze": "analysis"
            }
        )

        # Final edges
        workflow.add_edge("analysis", "response")
        workflow.add_edge("response", END)

        return workflow.compile()

    # ============ NODE IMPLEMENTATIONS ============

    async def _classify_node(self, state: CyberSecurityState) -> Dict[str, Any]:
        """
        Node 0: Classification

        Determines if the query is security-related or a general question.
        This prevents treating every query as a security investigation.

        Args:
            state: Current investigation state

        Returns:
            Updated state with classification
        """
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""

        # Quick heuristic check first (fast path)
        security_keywords = [
            "malware", "virus", "attack", "threat", "suspicious", "breach", "hack",
            "phishing", "ransomware", "exploit", "vulnerability", "ioc", "indicator",
            "ip", "domain", "hash", "forensic", "incident", "security", "log",
            "analyze", "investigate", "detect", "scan", "firewall", "intrusion"
        ]

        # If contains security keywords, mark as security query
        if any(keyword in last_message.lower() for keyword in security_keywords):
            return {**state, "investigation_steps": ["Classified as security query"]}

        # Otherwise, it's a general query
        return {**state, "investigation_steps": ["Classified as general query"]}

    async def _simple_response_node(self, state: CyberSecurityState) -> Dict[str, Any]:
        """
        Node: Simple Response

        Handles non-security queries with a straightforward response.

        Args:
            state: Current investigation state

        Returns:
            Updated state with simple response
        """
        messages = state["messages"]

        # Use LLM to respond naturally to general questions
        simple_prompt = """You are Aegis AI, a cybersecurity assistant.

The user has asked a general question (not related to security investigation).
Respond naturally and helpfully. If they want cybersecurity help, guide them on what they can ask."""

        response = await self.llm.ainvoke([
            SystemMessage(content=simple_prompt),
            *messages
        ])

        logger.info("Generated simple response for non-security query")

        return {
            **state,
            "messages": state["messages"] + [response],
            "final_response": response.content,
        }

    async def _planning_node(self, state: CyberSecurityState) -> Dict[str, Any]:
        """
        Node 1: Planning

        Assesses the user's query and determines the investigation approach.
        Performs initial risk classification and outlines investigation steps.

        Args:
            state: Current investigation state

        Returns:
            Updated state with investigation plan and severity assessment
        """
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""

        planning_prompt = """You are a cybersecurity investigation planner. Analyze this query and determine:

1. What type of investigation is needed? (log analysis, threat hunting, IOC lookup, malware analysis, incident response, etc.)
2. What information do we have vs. what we need to gather?
3. Initial risk assessment based on the query (low/medium/high/critical)
4. Recommended investigation approach (2-3 sentences)

Provide a brief, actionable investigation plan."""

        response = await self.llm.ainvoke([
            SystemMessage(content=planning_prompt),
            HumanMessage(content=last_message)
        ])

        # Extract severity from response (heuristic-based classification)
        severity = "medium"  # default
        content_lower = response.content.lower()

        if any(keyword in content_lower for keyword in ["critical", "urgent", "breach", "ransomware", "data exfiltration"]):
            severity = "critical"
        elif any(keyword in content_lower for keyword in ["high risk", "malware", "exploit", "compromise"]):
            severity = "high"
        elif any(keyword in content_lower for keyword in ["suspicious", "anomaly", "unusual"]):
            severity = "medium"
        elif any(keyword in content_lower for keyword in ["low risk", "informational", "benign"]):
            severity = "low"

        logger.info(f"Planning complete. Severity: {severity}")

        return {
            **state,
            "investigation_steps": [f"Planning: {response.content}"],
            "severity_level": severity,
            "messages": state["messages"] + [response],
        }

    async def _tool_selection_node(self, state: CyberSecurityState) -> Dict[str, Any]:
        """
        Node 2: Tool Selection

        LLM analyzes the investigation plan and selects appropriate tools.
        If no tools are available, proceeds to analysis with reasoning only.

        Args:
            state: Current investigation state

        Returns:
            Updated state with tool calls (if tools were selected)
        """
        messages = state["messages"]

        # If no tools available, skip this node
        if not self.tools:
            logger.info("No tools available - proceeding to analysis")
            return state

        # Bind tools to LLM so it can decide which to call
        llm_with_tools = self.llm.bind_tools(self.tools)

        tool_selection_prompt = """Based on the investigation plan, determine which tools would be most helpful.

Available tools can help with:
- IOC analysis (IP addresses, domains, file hashes)
- MITRE ATT&CK framework mapping
- Log parsing and analysis
- Threat intelligence lookups

Select appropriate tools or proceed without tools if reasoning alone is sufficient."""

        response = await llm_with_tools.ainvoke([
            SystemMessage(content=tool_selection_prompt),
            *messages
        ])

        # Check if tools were called
        tool_calls = getattr(response, 'tool_calls', [])
        if tool_calls:
            logger.info(f"Tools selected: {[tc['name'] for tc in tool_calls]}")
        else:
            logger.info("No tools selected - continuing with reasoning")

        return {
            **state,
            "messages": state["messages"] + [response],
        }

    async def _execute_tools_node(self, state: CyberSecurityState) -> Dict[str, Any]:
        """
        Node 3: Tool Execution

        Executes tools selected by the LLM and captures results.

        Args:
            state: Current investigation state

        Returns:
            Updated state with tool execution results
        """
        last_message = state["messages"][-1]

        # Check if there are tool calls to execute
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info(f"Executing {len(last_message.tool_calls)} tool(s)")

            # Use ToolNode to execute all tool calls
            result = await self.tool_node.ainvoke(state)

            # Track which tools were used
            tools_used = state.get("tools_used", [])
            tool_results = state.get("tool_results", [])

            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tools_used.append(tool_name)
                tool_results.append({
                    "tool": tool_name,
                    "args": tool_call.get("args", {}),
                    "timestamp": "now"  # You could add actual timestamp
                })

            investigation_steps = state.get("investigation_steps", [])
            investigation_steps.append(f"Executed tools: {', '.join([tc['name'] for tc in last_message.tool_calls])}")

            return {
                **result,
                "tools_used": tools_used,
                "tool_results": tool_results,
                "investigation_steps": investigation_steps,
            }

        # No tools to execute
        return state

    async def _analysis_node(self, state: CyberSecurityState) -> Dict[str, Any]:
        """
        Node 4: Analysis

        Correlates gathered information, identifies patterns, and assesses threats.
        Maps findings to MITRE ATT&CK framework where applicable.

        Args:
            state: Current investigation state

        Returns:
            Updated state with analysis and MITRE mappings
        """
        messages = state["messages"]
        iocs = state.get("iocs_found", [])
        severity = state.get("severity_level", "medium")
        tools_used = state.get("tools_used", [])

        analysis_prompt = f"""Analyze the investigation results and provide a comprehensive assessment:

**Investigation Context:**
- Severity Level: {severity}
- IOCs Detected: {len(iocs)}
- Tools Used: {', '.join(tools_used) if tools_used else 'None (reasoning-only)'}

**Your Analysis Should Include:**
1. Key Findings Summary (what was discovered?)
2. MITRE ATT&CK Tactics/Techniques (if applicable)
3. Threat Assessment (confidence level, potential impact)
4. Recommended Actions (immediate steps, further investigation needed)

Be specific and reference evidence from the investigation."""

        response = await self.llm.ainvoke([
            SystemMessage(content=analysis_prompt),
            *messages
        ])

        # Extract MITRE ATT&CK tactics using keyword matching
        mitre_tactics = []
        tactics_keywords = {
            "reconnaissance": "Reconnaissance (TA0043)",
            "initial access": "Initial Access (TA0001)",
            "execution": "Execution (TA0002)",
            "persistence": "Persistence (TA0003)",
            "privilege escalation": "Privilege Escalation (TA0004)",
            "defense evasion": "Defense Evasion (TA0005)",
            "credential access": "Credential Access (TA0006)",
            "discovery": "Discovery (TA0007)",
            "lateral movement": "Lateral Movement (TA0008)",
            "collection": "Collection (TA0009)",
            "exfiltration": "Exfiltration (TA0010)",
            "command and control": "Command and Control (TA0011)",
            "impact": "Impact (TA0040)",
        }

        content_lower = response.content.lower()
        for keyword, tactic in tactics_keywords.items():
            if keyword in content_lower:
                mitre_tactics.append(tactic)

        if mitre_tactics:
            logger.info(f"MITRE tactics identified: {mitre_tactics}")

        investigation_steps = state.get("investigation_steps", [])
        investigation_steps.append(f"Analysis: Identified {len(mitre_tactics)} MITRE tactics")

        return {
            **state,
            "messages": state["messages"] + [response],
            "mitre_tactics": mitre_tactics,
            "investigation_steps": investigation_steps,
        }

    async def _response_node(self, state: CyberSecurityState) -> Dict[str, Any]:
        """
        Node 5: Response Generation

        Generates a final formatted investigation report for the user.
        Structures findings in a clear, professional format suitable for SOC analysts.

        Args:
            state: Current investigation state

        Returns:
            Updated state with final formatted response
        """
        messages = state["messages"]
        severity = state.get("severity_level", "medium")
        mitre = state.get("mitre_tactics", [])
        iocs = state.get("iocs_found", [])
        tools_used = state.get("tools_used", [])

        response_prompt = f"""Generate a final cybersecurity investigation report based on the analysis.

**Investigation Summary:**
- Severity: {severity.upper()}
- MITRE ATT&CK Tactics: {', '.join(mitre) if mitre else 'None identified'}
- IOCs Detected: {len(iocs)}
- Tools Used: {', '.join(tools_used) if tools_used else 'Reasoning-only analysis'}

**Format Requirements:**
- Clear, professional tone suitable for a SOC analyst
- Structured sections (Executive Summary, Findings, Recommendations)
- Actionable recommendations
- Reference specific evidence

Generate the final report now."""

        final_response = await self.llm.ainvoke([
            SystemMessage(content=response_prompt),
            *messages
        ])

        logger.info("Investigation complete - final report generated")

        return {
            **state,
            "messages": state["messages"] + [final_response],
            "final_response": final_response.content,
        }

    # ============ CONDITIONAL EDGE FUNCTIONS ============

    def _is_security_query(self, state: CyberSecurityState) -> Literal["security", "general"]:
        """
        Determine if the query is security-related or general.

        Args:
            state: Current investigation state

        Returns:
            "security" if it's a cybersecurity query, "general" otherwise
        """
        investigation_steps = state.get("investigation_steps", [])

        # Check the classification from the classify node
        if investigation_steps and "security query" in investigation_steps[0].lower():
            return "security"
        return "general"

    def _should_use_tools(self, state: CyberSecurityState) -> Literal["execute", "skip"]:
        """
        Decide whether tools should be executed or skipped.

        Args:
            state: Current investigation state

        Returns:
            "execute" if tools were called, "skip" otherwise
        """
        last_message = state["messages"][-1]

        # Check if LLM requested tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "execute"
        return "skip"

    def _continue_investigation(self, state: CyberSecurityState) -> Literal["continue", "analyze"]:
        """
        Decide if more investigation steps are needed or if we should analyze.

        Checks:
        1. Have we exceeded max tool steps?
        2. Did the last message request more tool calls?

        Args:
            state: Current investigation state

        Returns:
            "continue" to loop back for more tools, "analyze" to proceed to analysis
        """
        # Check max tool iterations (prevent infinite loops)
        max_steps = getattr(settings, 'max_tool_steps', 5)
        tools_used = state.get("tools_used", [])

        if len(tools_used) >= max_steps:
            logger.info(f"Max tool steps ({max_steps}) reached - moving to analysis")
            return "analyze"

        # Check if LLM wants to call more tools
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info("Additional tools requested - continuing investigation")
            return "continue"

        return "analyze"

    # ============ STREAMING INTERFACE ============

    async def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat completion using LangGraph agent.

        This method implements the same interface as ClaudeClient and GeminiClient,
        allowing seamless integration with the existing gateway pattern.

        Args:
            model: Model identifier (e.g., "agent/cyber-analyst")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature (currently not used by graph)

        Yields:
            Chunks in OpenAI-compatible format:
            {"choices": [{"delta": {"content": "text"}}]}

        Example:
            async for chunk in agent.stream_chat_completion(
                model="agent/cyber-analyst",
                messages=[{"role": "user", "content": "Analyze IP 1.2.3.4"}],
                temperature=0.7
            ):
                print(chunk)
        """
        logger.info(f"Starting LangGraph agent execution with {len(messages)} messages")

        # Debug: log the messages structure
        logger.debug(f"Received messages: {messages}")

        # Convert messages to LangChain format
        lc_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # Handle content as list (multimodal support)
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                content = " ".join(text_parts)

            # Skip empty messages
            if not content or not role:
                logger.warning(f"Skipping empty message: role={role}, content={content}")
                continue

            # Convert to LangChain message types (skip system messages for graph)
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        logger.info(f"Converted to {len(lc_messages)} LangChain messages")

        # Ensure we have at least one message
        if not lc_messages:
            error_msg = "No valid messages received. Please provide a query."
            logger.error(f"{error_msg} Original messages: {messages}")
            yield {
                "choices": [{
                    "delta": {
                        "content": f"⚠️ **Error**: {error_msg}"
                    }
                }]
            }
            return

        # Initialize investigation state
        initial_state: CyberSecurityState = {
            "messages": lc_messages,
            "investigation_steps": [],
            "iocs_found": [],
            "severity_level": "medium",
            "mitre_tactics": [],
            "tools_used": [],
            "tool_results": [],
            "pending_approval": None,
            "final_response": "",
        }

        try:
            # Stream events from the graph execution
            async for event in self.app.astream_events(
                initial_state,
                version="v2"
            ):
                # Transform LangGraph events to OpenAI-compatible SSE format
                async for chunk in self._transform_event_to_sse(event):
                    yield chunk

        except Exception as e:
            logger.error(f"LangGraph agent error: {type(e).__name__}: {str(e)}", exc_info=True)
            # Yield error as SSE event
            yield {
                "choices": [{
                    "delta": {
                        "content": f"\n\n⚠️ **Error during investigation**: {str(e)}"
                    }
                }]
            }

    async def _transform_event_to_sse(
        self,
        event: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Transform LangGraph stream events into OpenAI-compatible SSE chunks.

        This adapter ensures compatibility with the existing frontend that expects
        OpenAI's streaming format.

        LangGraph Event Types:
        - on_chat_model_stream: LLM generating text
        - on_tool_start: Tool execution beginning
        - on_tool_end: Tool execution complete
        - on_chain_start: Node execution beginning
        - on_chain_end: Node execution complete

        OpenAI SSE Format:
        {
            "choices": [{
                "delta": {"content": "text content"}
            }]
        }

        Args:
            event: LangGraph stream event

        Yields:
            OpenAI-compatible chunk dicts
        """
        event_type = event.get("event")
        data = event.get("data", {})
        name = event.get("name", "")

        # Stream LLM text generation
        if event_type == "on_chat_model_stream":
            chunk = data.get("chunk")
            if chunk and hasattr(chunk, 'content') and chunk.content:
                yield {
                    "choices": [{
                        "delta": {
                            "content": chunk.content
                        }
                    }]
                }

        # Stream tool execution updates
        elif event_type == "on_tool_start":
            tool_input = data.get("input", {})
            tool_name = tool_input.get("name", name)
            yield {
                "choices": [{
                    "delta": {
                        "content": f"\n\n🔧 **Using tool**: `{tool_name}`\n"
                    }
                }]
            }

        elif event_type == "on_tool_end":
            yield {
                "choices": [{
                    "delta": {
                        "content": " ✓\n"
                    }
                }]
            }

        # Stream node transitions (investigation progress indicators)
        elif event_type == "on_chain_start":
            # Add headers for major investigation phases
            if "planning" in name.lower():
                yield {
                    "choices": [{
                        "delta": {
                            "content": "# 🔍 Investigation Planning\n\n"
                        }
                    }]
                }
            elif "tool_selection" in name.lower():
                yield {
                    "choices": [{
                        "delta": {
                            "content": "\n\n# 🛠️ Tool Selection\n\n"
                        }
                    }]
                }
            elif "analysis" in name.lower():
                yield {
                    "choices": [{
                        "delta": {
                            "content": "\n\n# 📊 Threat Analysis\n\n"
                        }
                    }]
                }
            elif "response" in name.lower():
                yield {
                    "choices": [{
                        "delta": {
                            "content": "\n\n# 📋 Investigation Report\n\n"
                        }
                    }]
                }
