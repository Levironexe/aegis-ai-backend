# Provider Icon Live Update Fix

## Problem
When users changed the AI model in the chat interface, the provider icons (Anthropic, OpenAI, Google, etc.) for messages would not update correctly:
- During streaming, icons would show the old model
- After streaming finished, icons would revert to Claude
- When changing models, all previous messages would incorrectly change to the new model's icon

## Solution Overview
Implemented a two-layer provider tracking system:
1. **Database layer**: Store provider in `Message_v2` table for persistence
2. **Client layer**: Inject provider into message metadata during streaming for immediate UI updates

## Changes Made

### Backend Changes

#### 1. Database Schema (`app/models/chat.py`)
Added `provider` column to Message model:
```python
class Message(Base):
    __tablename__ = "Message_v2"
    # ... existing fields ...
    provider = Column(String(20), nullable=True)  # NEW: Model provider
```

#### 2. API Response Schema (`app/schemas/chat.py`)
Added provider to MessageResponse:
```python
class MessageResponse(BaseModel):
    # ... existing fields ...
    provider: Optional[str] = None  # NEW
```

#### 3. Message Creation (`app/routers/chat.py`)
Extract and save provider when creating assistant messages:
```python
# Extract provider from model string (e.g., "anthropic/claude-sonnet-4.5" -> "anthropic")
provider = model.split("/")[0] if "/" in model else None

assistant_message = Message(
    chatId=chat_id,
    role="assistant",
    parts=[{"type": "text", "text": full_content}],
    attachments=[],
    provider=provider,  # NEW
    createdAt=datetime.utcnow()
)
```

#### 4. Database Migration
Created migration script to add `provider` column:
```python
await conn.execute('''
    ALTER TABLE "Message_v2"
    ADD COLUMN provider VARCHAR(20) NULL;
''')
```

### Frontend Changes

#### 1. Database Schema (`lib/db/schema.ts`)
Added provider column to frontend schema:
```typescript
export const message = pgTable("Message_v2", {
  // ... existing fields ...
  provider: varchar("provider", { length: 20 }),
});
```

#### 2. Type Definitions (`lib/types.ts`)
Updated metadata schema to include provider:
```typescript
export const messageMetadataSchema = z.object({
  createdAt: z.string(),
  provider: z.string().optional(),  // NEW
});
```

#### 3. Message Conversion (`lib/utils.ts`)
Map provider from database to message metadata:
```typescript
export function convertToUIMessages(messages: DBMessage[]): ChatMessage[] {
  return messages.map((message) => ({
    id: message.id,
    role: message.role,
    parts: message.parts,
    metadata: {
      createdAt: formatISO(message.createdAt),
      provider: message.provider || undefined,  // NEW
    },
  }));
}
```

#### 4. Chat Component (`components/chat.tsx`)
**Key Fix #1**: Pass current model instead of initial model to Messages component:
```typescript
<Messages
  selectedModelId={currentModelId}  // Changed from initialChatModel
  // ... other props
/>
```

**Key Fix #2**: Inject provider into message metadata when streaming finishes:
```typescript
onFinish: () => {
  // Inject provider into the last assistant message's metadata
  const provider = currentModelIdRef.current?.split("/")[0] || null;
  if (provider) {
    setMessages((currentMessages) => {
      const lastMessage = currentMessages[currentMessages.length - 1];
      if (lastMessage?.role === "assistant" && !lastMessage.metadata?.provider) {
        return currentMessages.map((msg, idx) => {
          if (idx === currentMessages.length - 1) {
            return {
              ...msg,
              metadata: {
                createdAt: msg.metadata?.createdAt || new Date().toISOString(),
                provider,  // Inject provider immediately
              },
            } as ChatMessage;
          }
          return msg;
        });
      }
      return currentMessages;
    });
  }
  mutate(unstable_serialize(getChatHistoryPaginationKey));
}
```

#### 5. Message Display (`components/message.tsx`)
Use provider from metadata first, fall back to selectedModelId for new messages:
```typescript
// Priority: 1) message metadata provider (from DB), 2) selectedModelId (for streaming/new messages)
const provider = message.metadata?.provider?.toLowerCase() ||
  (() => {
    const selectedModel = chatModels.find((model) => model.id === selectedModelId);
    return selectedModel?.provider.toLowerCase() || "anthropic";
  })();
```

## How It Works

### Message Lifecycle
1. **User changes model**: `currentModelId` state updates → `currentModelIdRef.current` updates
2. **User sends message**: API request uses `currentModelIdRef.current` for model selection
3. **Streaming starts**: Message component uses `selectedModelId` prop (which is `currentModelId`) to display icon
4. **Streaming finishes**: `onFinish` callback injects provider into `message.metadata.provider`
5. **After streaming**: Message now has metadata.provider, won't change even if user changes model selector
6. **Backend saves**: Provider saved to database `Message_v2.provider` column
7. **Page refresh**: Messages load from DB with provider in metadata

### Provider Resolution Logic
For each message, the UI determines which provider icon to show:
```
1. Does message.metadata.provider exist?
   YES → Use it (message was saved or just finished streaming)
   NO  → Use selectedModelId to determine provider (message is actively streaming)
```

## Benefits
✅ **Live Updates**: Icons update immediately when model changes
✅ **Persistence**: Icons persist after page refresh (database)
✅ **Correctness**: Each message shows the icon of the model that actually generated it
✅ **No Flicker**: Smooth transition from streaming → finished → persisted

## Testing Checklist
- [ ] Change model and send message - icon shows correct provider during streaming
- [ ] Message finishes streaming - icon stays correct (doesn't revert to default)
- [ ] Change model selector after sending - previous messages keep their original icons
- [ ] Refresh page - all messages show correct icons from database
- [ ] Send messages with different models - each shows its own provider icon
