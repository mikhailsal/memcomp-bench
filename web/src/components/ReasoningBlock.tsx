import { useState } from 'react';

interface ReasoningBlockProps {
  text: string;
  label?: string;
  variant?: 'thinking' | 'monologue' | 'content' | 'human';
}

const VARIANT_STYLES = {
  thinking: { border: 'border-violet-500/20', bg: 'bg-violet-950/20', text: 'text-violet-400', content: 'text-violet-300/80' },
  monologue: { border: 'border-amber-500/20', bg: 'bg-amber-950/20', text: 'text-amber-400', content: 'text-amber-300/80' },
  content: { border: 'border-slate-500/20', bg: 'bg-slate-800/40', text: 'text-slate-400', content: 'text-slate-300/80' },
  human: { border: 'border-blue-500/20', bg: 'bg-blue-950/20', text: 'text-blue-400', content: 'text-blue-300/80' },
};

export default function ReasoningBlock({ text, label, variant = 'thinking' }: ReasoningBlockProps) {
  const [expanded, setExpanded] = useState(false);
  const style = VARIANT_STYLES[variant];

  return (
    <div className={`rounded-lg border ${style.border} ${style.bg} overflow-hidden`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className={`w-full flex items-center gap-2 px-3 py-2 text-xs ${style.text} hover:opacity-80 transition-opacity`}
      >
        <svg className={`w-3.5 h-3.5 transition-transform ${expanded ? 'rotate-90' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
        <span className="font-semibold uppercase tracking-wide">{label ?? 'Thinking'}</span>
      </button>
      {expanded && (
        <div className={`px-3 pb-3 text-xs ${style.content} leading-relaxed whitespace-pre-wrap border-t ${style.border}`}>
          {text}
        </div>
      )}
    </div>
  );
}
