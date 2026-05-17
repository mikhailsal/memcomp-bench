import { Link } from 'react-router-dom';
import type { ConversationSummary } from '../lib/types';
import { formatCost, formatTokens, relativeTime, langFlag } from '../lib/formatters';

interface ConversationCardProps {
  conversation: ConversationSummary;
}

export default function ConversationCard({ conversation: c }: ConversationCardProps) {
  return (
    <Link
      to={`/conversation/${c.id}`}
      className="block p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-raised)] hover:border-sky-500/40 hover:bg-[var(--color-surface-hover)] transition-all group"
    >
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2">
          <span className="text-lg">{'\uD83D\uDC64'}</span>
          <h3 className="font-semibold text-[var(--color-text)] group-hover:text-sky-400 transition-colors">
            {c.profile}
          </h3>
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/20 border border-white/5 text-[var(--color-text-muted)]">
            {langFlag(c.language)}
          </span>
        </div>
        <span className="text-xs text-[var(--color-text-muted)] whitespace-nowrap">
          {relativeTime(c.startedAt)}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-[var(--color-text-muted)] mb-3">
        <div>AI: <span className="font-mono text-[var(--color-text)]">{c.aiModelShort}</span></div>
        <div>Human: <span className="font-mono text-[var(--color-text)]">{c.humanModelShort}</span></div>
      </div>

      <div className="flex items-center gap-3 text-xs text-[var(--color-text-muted)]">
        <span>{c.turns} turns</span>
        <span className="text-[var(--color-border)]">&middot;</span>
        <span>{formatTokens(c.tokens)} tok</span>
        <span className="text-[var(--color-border)]">&middot;</span>
        <span>{formatCost(c.cost)}</span>
        {c.seedWords.length > 0 && (
          <>
            <span className="text-[var(--color-border)]">&middot;</span>
            <span className="truncate max-w-[120px]" title={c.seedWords.join(', ')}>
              {c.seedWords.slice(0, 3).join(', ')}
            </span>
          </>
        )}
      </div>
    </Link>
  );
}
