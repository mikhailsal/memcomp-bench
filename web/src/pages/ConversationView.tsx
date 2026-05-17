import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useConversation } from '../lib/manifest';
import { fetchConversation, extractToolCallReasoning } from '../lib/parseConversation';
import { renderMarkdown } from '../lib/markdown';
import type { ParsedConversation, ConversationTurn, ConversationEvent } from '../lib/types';
import { formatTokens } from '../lib/formatters';
import ChatBubble from '../components/ChatBubble';
import ReasoningBlock from '../components/ReasoningBlock';
import MetadataPanel from '../components/MetadataPanel';
import EventMarker from '../components/EventMarker';

function MessageContent({ text }: { text: string }) {
  return (
    <div
      className="text-sm leading-relaxed break-words prose-invert"
      dangerouslySetInnerHTML={{ __html: renderMarkdown(text) }}
    />
  );
}

function TurnTokenBadge({ aiCtx, humanCtx }: { aiCtx: number; humanCtx: number }) {
  if (!aiCtx && !humanCtx) return null;
  return (
    <div className="text-[10px] text-[var(--color-text-muted)] font-mono mt-1">
      AI ctx: {formatTokens(aiCtx)} &middot; Human ctx: {formatTokens(humanCtx)}
    </div>
  );
}

function renderTurn(turn: ConversationTurn) {
  const isHuman = turn.speaker === 'human';
  const toolReasoning = !isHuman ? extractToolCallReasoning(turn) : null;

  return (
    <div key={`turn-${turn.turn_number}`} className="space-y-2">
      {/* Human reasoning (rare but possible) */}
      {isHuman && turn.human_reasoning && (
        <ReasoningBlock
          text={turn.human_reasoning}
          label="Human Simulator Reasoning"
          variant="human"
        />
      )}

      {/* AI native reasoning */}
      {!isHuman && turn.ai_reasoning && (
        <ReasoningBlock
          text={turn.ai_reasoning}
          label="Native Reasoning"
          variant="thinking"
        />
      )}

      {/* AI inner monologue from tool call */}
      {!isHuman && toolReasoning && toolReasoning !== turn.ai_reasoning && (
        <ReasoningBlock
          text={toolReasoning}
          label="Inner Monologue"
          variant="monologue"
        />
      )}

      {/* AI thinking fallback */}
      {!isHuman && turn.ai_thinking && turn.ai_thinking !== turn.ai_reasoning && turn.ai_thinking !== toolReasoning && (
        <ReasoningBlock
          text={turn.ai_thinking}
          label="Thinking"
          variant="thinking"
        />
      )}

      {/* AI content draft */}
      {!isHuman && turn.ai_content && turn.ai_content !== turn.ai_reasoning && turn.ai_content !== toolReasoning && (
        <ReasoningBlock
          text={turn.ai_content}
          label="Response Draft"
          variant="content"
        />
      )}

      <ChatBubble
        type={isHuman ? 'human' : 'ai'}
        label={isHuman ? turn.speaker === 'human' ? 'Human' : 'AI' : 'AI'}
        badge={`turn ${turn.turn_number}`}
      >
        <MessageContent text={turn.visible_text} />
        <TurnTokenBadge aiCtx={turn.ai_context_tokens} humanCtx={turn.human_context_tokens} />
      </ChatBubble>
    </div>
  );
}

export default function ConversationView() {
  const { id } = useParams<{ id: string }>();
  const summary = useConversation(id);
  const [data, setData] = useState<ParsedConversation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showPlan, setShowPlan] = useState(false);
  const [showBackstory, setShowBackstory] = useState(false);

  useEffect(() => {
    if (!summary) return;
    setLoading(true);
    setError(null);
    fetchConversation(summary.fileName)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [summary]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-[var(--color-text-muted)]">Loading conversation...</div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <p className="text-red-400">{error ?? 'Conversation not found'}</p>
        <Link to="/" className="text-sky-400 hover:underline text-sm">Back to conversations</Link>
      </div>
    );
  }

  const { metadata, turns, events } = data;

  const eventsByTurn = new Map<number, ConversationEvent[]>();
  for (const e of events) {
    const arr = eventsByTurn.get(e.turn_number) ?? [];
    arr.push(e);
    eventsByTurn.set(e.turn_number, arr);
  }

  return (
    <div className="animate-fade-in max-w-4xl mx-auto">
      {/* Breadcrumb */}
      <div className="mb-4 text-sm flex items-center gap-1">
        <Link to="/" className="text-[var(--color-text-muted)] hover:text-sky-400 transition-colors">Conversations</Link>
        <span className="text-[var(--color-text-muted)]">/</span>
        <span className="font-mono">{metadata.human_profile.name} &mdash; {metadata.conversation_id}</span>
      </div>

      <MetadataPanel metadata={metadata} />

      {/* Backstory toggle */}
      <div className="mb-4">
        <button
          onClick={() => setShowBackstory(!showBackstory)}
          className="flex items-center gap-2 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors"
        >
          <svg className={`w-3.5 h-3.5 transition-transform ${showBackstory ? 'rotate-90' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <span className="font-semibold uppercase tracking-wide">Human Profile Backstory</span>
        </button>
        {showBackstory && (
          <div className="mt-2 p-3 rounded-lg bg-[var(--color-surface-raised)] border border-[var(--color-border)] text-sm text-[var(--color-text-muted)] leading-relaxed">
            {metadata.human_profile.backstory}
          </div>
        )}
      </div>

      {/* Conversation plan toggle */}
      {metadata.conversation_plan && (
        <div className="mb-6">
          <button
            onClick={() => setShowPlan(!showPlan)}
            className="flex items-center gap-2 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors"
          >
            <svg className={`w-3.5 h-3.5 transition-transform ${showPlan ? 'rotate-90' : ''}`}
              fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
            <span className="font-semibold uppercase tracking-wide">Conversation Plan</span>
          </button>
          {showPlan && (
            <div className="mt-2 p-3 rounded-lg bg-[var(--color-surface-raised)] border border-[var(--color-border)] text-sm text-[var(--color-text-muted)] leading-relaxed whitespace-pre-wrap">
              {metadata.conversation_plan}
            </div>
          )}
        </div>
      )}

      {/* Conversation turns */}
      <div className="space-y-4">
        {turns.map(turn => {
          const eventsBeforeTurn = eventsByTurn.get(turn.turn_number);
          return (
            <div key={`group-${turn.turn_number}`}>
              {eventsBeforeTurn?.map((evt, i) => (
                <EventMarker key={`event-${turn.turn_number}-${i}`} event={evt} />
              ))}
              {renderTurn(turn)}
            </div>
          );
        })}
      </div>

      {/* Back link */}
      <div className="mt-8 mb-4 text-center">
        <Link to="/" className="text-sky-400 hover:underline text-sm">
          &larr; Back to conversations
        </Link>
      </div>
    </div>
  );
}
