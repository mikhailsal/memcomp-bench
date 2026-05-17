import type { ParsedConversation, ConversationMetadata, ConversationTurn, ConversationEvent } from './types';

export function parseConversation(text: string): ParsedConversation {
  const lines = text.split('\n').filter(l => l.trim());
  let metadata: ConversationMetadata | null = null;
  const turns: ConversationTurn[] = [];
  const events: ConversationEvent[] = [];

  for (const line of lines) {
    const obj = JSON.parse(line);
    switch (obj.type) {
      case 'metadata':
        metadata = obj as ConversationMetadata;
        break;
      case 'turn':
        turns.push(obj as ConversationTurn);
        break;
      case 'event':
        events.push(obj as ConversationEvent);
        break;
    }
  }

  if (!metadata) throw new Error('No metadata found in conversation file');

  return { metadata, turns, events };
}

export function fetchConversation(fileName: string): Promise<ParsedConversation> {
  const base = import.meta.env.BASE_URL;
  return fetch(`${base}output/${fileName}`)
    .then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.text();
    })
    .then(parseConversation);
}

export function extractToolCallReasoning(turn: ConversationTurn): string | null {
  if (!turn.ai_tool_calls?.length) return null;
  const tc = turn.ai_tool_calls[0];
  try {
    const parsed = JSON.parse(tc.function.arguments);
    return parsed.reasoning || null;
  } catch {
    return null;
  }
}
