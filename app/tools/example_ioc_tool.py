"""
Example IOC Analysis Tool (Specimen)

This is a specimen/example tool to demonstrate how to create tools for the LangGraph agent.
It provides a template for building real cybersecurity tools.

IMPORTANT: This is NOT a real tool - it returns mock data for demonstration purposes only.
You should replace this with actual IOC analysis logic (e.g., VirusTotal API, MISP, etc.)

Usage:
    from app.tools.example_ioc_tool import ExampleIOCTool

    tool = ExampleIOCTool()
    result = await tool.execute(indicator="192.168.1.100", indicator_type="ip")
    langchain_tool = tool.to_langchain_tool()  # Convert for LangGraph
"""

from app.tools.base import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal


class IOCAnalysisInput(BaseModel):
    """
    Input schema for IOC analysis tool.

    Uses Pydantic for type validation and documentation.
    The LLM will see these field descriptions when selecting tools.
    """
    indicator: str = Field(
        description="The indicator to analyze (IP address, domain, file hash, etc.)"
    )
    indicator_type: Literal["ip", "domain", "hash", "url"] = Field(
        description="Type of indicator: 'ip', 'domain', 'hash', or 'url'"
    )


class ExampleIOCTool(BaseTool):
    """
    Example tool for analyzing Indicators of Compromise (IOCs).

    This is a specimen tool that demonstrates:
    1. How to extend BaseTool
    2. How to define input schema with Pydantic
    3. How to implement async execute method
    4. How to return structured results

    Real Implementation Ideas:
    - Integrate with VirusTotal API for reputation checks
    - Query MISP for threat intelligence
    - Check against known malicious IP/domain lists
    - Analyze file hashes against malware databases
    - Perform WHOIS lookups for domains/IPs
    """

    def __init__(self):
        super().__init__()
        # Override default name if needed
        self.name = "analyze_ioc"
        # Set to True if this tool requires human approval
        self.needs_approval = False

    @property
    def description(self) -> str:
        """
        Description shown to the LLM when selecting tools.

        Make this clear and specific so the LLM knows when to use this tool.
        """
        return (
            "Analyzes Indicators of Compromise (IOCs) such as IP addresses, domains, "
            "file hashes, and URLs. Returns threat intelligence, reputation score, "
            "and recommendations. Use this tool when you need to assess whether an "
            "indicator is malicious or benign."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        """Return the Pydantic model for input validation."""
        return IOCAnalysisInput

    async def execute(self, indicator: str, indicator_type: str) -> Dict[str, Any]:
        """
        Execute IOC analysis.

        This is a mock implementation for demonstration. Replace with real logic.

        Args:
            indicator: The IOC to analyze (e.g., "192.168.1.100", "evil.com")
            indicator_type: Type of indicator ("ip", "domain", "hash", "url")

        Returns:
            Dict containing analysis results:
            {
                "indicator": str,
                "type": str,
                "reputation": str,  # "malicious", "suspicious", "benign", "unknown"
                "confidence": float,  # 0.0 to 1.0
                "threat_categories": list[str],
                "details": str,
                "recommendations": str
            }
        """

        # MOCK IMPLEMENTATION - Replace with real IOC analysis
        # In a real tool, you would:
        # 1. Query threat intelligence APIs (VirusTotal, AbuseIPDB, etc.)
        # 2. Check internal threat databases
        # 3. Perform reputation lookups
        # 4. Analyze patterns and context

        # For demonstration, we'll return mock data based on simple heuristics
        mock_results = self._generate_mock_analysis(indicator, indicator_type)

        return mock_results

    def _generate_mock_analysis(self, indicator: str, indicator_type: str) -> Dict[str, Any]:
        """
        Generate mock analysis results for demonstration.

        This simulates what a real IOC analysis tool would return.
        """

        # Simple heuristics for demo purposes
        is_private_ip = False
        is_suspicious = False

        if indicator_type == "ip":
            # Check if it's a private IP (RFC 1918)
            if indicator.startswith(("10.", "172.16.", "192.168.")):
                is_private_ip = True
            # Simple "malicious" heuristic for demo
            elif any(char in indicator for char in ["1.1.1", "8.8.8", "255.255"]):
                is_suspicious = False  # Known good DNS servers
            else:
                is_suspicious = True  # Treat others as suspicious for demo

        elif indicator_type == "domain":
            # Simple heuristic: short domains or certain TLDs
            suspicious_tlds = [".xyz", ".tk", ".ml", ".ga"]
            if any(indicator.endswith(tld) for tld in suspicious_tlds):
                is_suspicious = True
            elif len(indicator.split(".")[0]) < 4:
                is_suspicious = True

        elif indicator_type == "hash":
            # For hashes, just demonstrate the pattern
            is_suspicious = len(indicator) == 32 or len(indicator) == 64

        elif indicator_type == "url":
            is_suspicious = "http://" in indicator or "bit.ly" in indicator

        # Generate mock results
        if is_private_ip:
            return {
                "indicator": indicator,
                "type": indicator_type,
                "reputation": "benign",
                "confidence": 0.95,
                "threat_categories": [],
                "details": (
                    f"The IP address {indicator} is a private (RFC 1918) address used for internal networking. "
                    "It cannot be directly accessed from the internet and is not inherently malicious."
                ),
                "recommendations": (
                    "No action required for private IP addresses. If this appears in logs, verify it matches "
                    "your expected internal network topology."
                ),
                "source": "Mock Analysis (Replace with real IOC service)"
            }

        elif is_suspicious:
            return {
                "indicator": indicator,
                "type": indicator_type,
                "reputation": "suspicious",
                "confidence": 0.75,
                "threat_categories": ["potential_threat", "requires_investigation"],
                "details": (
                    f"The {indicator_type} '{indicator}' exhibits characteristics commonly associated with "
                    "malicious infrastructure. Multiple threat intelligence sources have flagged similar indicators."
                ),
                "recommendations": (
                    "RECOMMENDED ACTIONS:\n"
                    "1. Block this indicator at network perimeter\n"
                    "2. Search logs for any connections to this indicator\n"
                    "3. Investigate any systems that communicated with it\n"
                    "4. Consider escalating to incident response team"
                ),
                "source": "Mock Analysis (Replace with real IOC service)"
            }

        else:
            return {
                "indicator": indicator,
                "type": indicator_type,
                "reputation": "benign",
                "confidence": 0.80,
                "threat_categories": [],
                "details": (
                    f"The {indicator_type} '{indicator}' appears to be benign based on current threat intelligence. "
                    "No malicious activity associated with this indicator has been detected."
                ),
                "recommendations": (
                    "No immediate action required. Continue monitoring for any changes in reputation."
                ),
                "source": "Mock Analysis (Replace with real IOC service)"
            }


# ============ USAGE EXAMPLE ============

async def example_usage():
    """
    Example of how to use this tool directly or with LangGraph.
    """

    # Direct usage
    tool = ExampleIOCTool()

    # Analyze an IP address
    result = await tool.execute(
        indicator="192.168.1.100",
        indicator_type="ip"
    )
    print(f"Direct call result: {result}")

    # Convert to LangChain tool for use with LangGraph
    langchain_tool = tool.to_langchain_tool()
    print(f"LangChain tool name: {langchain_tool.name}")
    print(f"LangChain tool description: {langchain_tool.description}")

    # The LangGraph agent can now use this tool automatically!


# ============ NOTES FOR BUILDING REAL TOOLS ============

"""
To create a real IOC analysis tool, you would:

1. **Choose Your Data Sources:**
   - VirusTotal API (https://www.virustotal.com/api/v3/)
   - AbuseIPDB (https://www.abuseipdb.com/)
   - MISP (Malware Information Sharing Platform)
   - AlienVault OTX (Open Threat Exchange)
   - Your organization's internal threat intel

2. **Implement API Calls:**
   ```python
   import httpx

   async def query_virustotal(indicator: str, indicator_type: str):
       api_key = settings.virustotal_api_key
       async with httpx.AsyncClient() as client:
           if indicator_type == "ip":
               url = f"https://www.virustotal.com/api/v3/ip_addresses/{indicator}"
           elif indicator_type == "domain":
               url = f"https://www.virustotal.com/api/v3/domains/{indicator}"
           # ... etc

           response = await client.get(url, headers={"x-apikey": api_key})
           return response.json()
   ```

3. **Aggregate Results:**
   - Combine data from multiple sources
   - Calculate confidence scores
   - Provide contextual recommendations

4. **Error Handling:**
   - Handle API rate limits
   - Handle network errors
   - Return graceful fallbacks

5. **Caching:**
   - Cache results to avoid redundant API calls
   - Use Redis or in-memory cache
   - Set appropriate TTLs

6. **Security:**
   - Never expose API keys in responses
   - Validate input to prevent injection
   - Rate limit tool usage

Example real implementation structure:

class RealIOCTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.vt_client = VirusTotalClient(api_key=settings.vt_api_key)
        self.abuseipdb_client = AbuseIPDBClient(api_key=settings.abuseipdb_key)
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache

    async def execute(self, indicator: str, indicator_type: str):
        # Check cache first
        cache_key = f"{indicator_type}:{indicator}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Query multiple sources in parallel
        results = await asyncio.gather(
            self._query_virustotal(indicator, indicator_type),
            self._query_abuseipdb(indicator, indicator_type),
            self._query_internal_db(indicator, indicator_type),
            return_exceptions=True
        )

        # Aggregate and analyze
        analysis = self._aggregate_results(results)

        # Cache result
        self.cache[cache_key] = analysis

        return analysis
"""
