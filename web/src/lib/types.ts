export interface ConversationSummary {
  id: string;
  fileName: string;
  profile: string;
  profileBackstory: string;
  aiModel: string;
  aiModelShort: string;
  humanModel: string;
  humanModelShort: string;
  language: string;
  turns: number;
  tokens: number;
  cost: number;
  seedWords: string[];
  startedAt: string;
  finishedAt: string;
  aiTemperature?: number;
  humanTemperature?: number;
  aiMaxTokens?: number;
  humanMaxTokens?: number;
  companionMode: string;
}

export interface Manifest {
  generatedAt: string;
  totalConversations: number;
  totalTurns: number;
  totalCost: number;
  profiles: string[];
  aiModels: string[];
  humanModels: string[];
  conversations: ConversationSummary[];
}

export interface ConversationTurn {
  type: 'turn';
  turn_number: number;
  speaker: 'human' | 'ai';
  visible_text: string;
  ai_thinking: string | null;
  ai_content: string | null;
  ai_reasoning: string | null;
  ai_tool_calls: ToolCall[] | null;
  ai_reasoning_details: ReasoningDetail[] | null;
  human_reasoning: string | null;
  human_reasoning_details: ReasoningDetail[] | null;
  token_estimate: number;
  cost_usd: number;
  timestamp: string;
  ai_context_tokens: number;
  human_context_tokens: number;
}

export interface ToolCall {
  type: string;
  index: number;
  id: string;
  function: {
    name: string;
    arguments: string;
  };
}

export interface ReasoningDetail {
  type: string;
  summary?: string;
  id?: string;
}

export interface ConversationEvent {
  type: 'event';
  event_type: string;
  turn_number: number;
  source: string;
  timestamp: string;
  message: string | null;
  previous_topic: string | null;
  current_topic: string | null;
  topic_changed: boolean | null;
  nudge_injected: boolean | null;
  suppression_reason: string | null;
}

export interface ConversationMetadata {
  type: 'metadata';
  conversation_id: string;
  human_profile: {
    name: string;
    backstory: string;
  };
  ai_model: string;
  human_model: string;
  seed_words: string[];
  conversation_plan: string;
  language: string;
  companion_mode: string;
  ai_provider: unknown;
  ai_reasoning: unknown;
  ai_temperature: number;
  ai_max_tokens: number;
  ai_rpm_limit: number | null;
  human_provider: unknown;
  human_reasoning: unknown;
  human_temperature: number;
  human_max_tokens: number;
  human_rpm_limit: number | null;
  total_turns: number;
  total_tokens_estimate: number;
  total_cost_usd: number;
  started_at: string;
  finished_at: string;
}

export interface ParsedConversation {
  metadata: ConversationMetadata;
  turns: ConversationTurn[];
  events: ConversationEvent[];
}
