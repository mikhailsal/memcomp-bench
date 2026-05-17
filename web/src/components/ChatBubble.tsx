import { useState } from 'react';

type BubbleType = 'human' | 'ai' | 'system' | 'event';

interface ChatBubbleProps {
  type: BubbleType;
  label: string;
  sublabel?: string;
  badge?: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  collapsible?: boolean;
}

const STYLE_MAP: Record<BubbleType, { bg: string; border: string; label: string; icon: string }> = {
  human: {
    bg: 'bg-blue-950/40',
    border: 'border-blue-500/20',
    label: 'text-blue-400',
    icon: '\uD83D\uDC64',
  },
  ai: {
    bg: 'bg-emerald-950/30',
    border: 'border-emerald-500/20',
    label: 'text-emerald-400',
    icon: '\uD83E\uDD16',
  },
  system: {
    bg: 'bg-slate-800/60',
    border: 'border-slate-600/30',
    label: 'text-slate-400',
    icon: '\u2699\uFE0F',
  },
  event: {
    bg: 'bg-amber-950/30',
    border: 'border-amber-500/20',
    label: 'text-amber-400',
    icon: '\uD83D\uDD14',
  },
};

export default function ChatBubble({
  type,
  label,
  sublabel,
  badge,
  children,
  defaultExpanded = true,
  collapsible = false,
}: ChatBubbleProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const style = STYLE_MAP[type];

  return (
    <div className={`rounded-xl border ${style.bg} ${style.border} overflow-hidden animate-fade-in`}>
      <div
        className={`px-4 py-2 flex items-center justify-between ${collapsible ? 'cursor-pointer hover:opacity-80' : ''}`}
        onClick={collapsible ? () => setExpanded(!expanded) : undefined}
      >
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm">{style.icon}</span>
          <span className={`text-xs font-semibold uppercase tracking-wide ${style.label}`}>{label}</span>
          {badge && (
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/20 border border-white/5 text-[var(--color-text-muted)]">
              {badge}
            </span>
          )}
          {sublabel && (
            <span className="text-xs text-[var(--color-text-muted)]">{sublabel}</span>
          )}
        </div>
        {collapsible && (
          <svg
            className={`w-4 h-4 text-[var(--color-text-muted)] transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        )}
      </div>

      {expanded && (
        <div className="px-4 pb-4">
          {children}
        </div>
      )}
    </div>
  );
}
