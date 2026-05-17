import type { ConversationMetadata } from '../lib/types';
import { formatCost, formatTokens, formatDateTime, formatDuration, langFlag } from '../lib/formatters';

interface MetadataPanelProps {
  metadata: ConversationMetadata;
}

export default function MetadataPanel({ metadata: m }: MetadataPanelProps) {
  return (
    <div className="p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)] mb-6 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 mb-4">
        <div>
          <h1 className="text-xl font-bold mb-1">
            {m.human_profile.name} & AI
          </h1>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-400 border border-sky-500/20">
              {langFlag(m.language)}
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/20">
              {m.companion_mode}
            </span>
          </div>
        </div>
        <div className="text-right text-xs text-[var(--color-text-muted)] space-y-0.5 shrink-0">
          <div>{formatDateTime(m.started_at)}</div>
          <div>Duration: {formatDuration(m.started_at, m.finished_at)}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        <StatBox label="Turns" value={String(m.total_turns)} />
        <StatBox label="Tokens" value={formatTokens(m.total_tokens_estimate)} />
        <StatBox label="Cost" value={formatCost(m.total_cost_usd)} />
        <StatBox label="Seed" value={m.seed_words.join(', ')} />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <ModelBox
          role="AI Companion"
          model={m.ai_model}
          temperature={m.ai_temperature}
          maxTokens={m.ai_max_tokens}
        />
        <ModelBox
          role="Human Simulator"
          model={m.human_model}
          temperature={m.human_temperature}
          maxTokens={m.human_max_tokens}
        />
      </div>
    </div>
  );
}

function StatBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-2.5 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)]">
      <div className="text-[10px] text-[var(--color-text-muted)] uppercase tracking-wide mb-0.5">{label}</div>
      <div className="text-sm font-semibold truncate" title={value}>{value}</div>
    </div>
  );
}

function ModelBox({ role, model, temperature, maxTokens }: {
  role: string;
  model: string;
  temperature: number;
  maxTokens: number;
}) {
  return (
    <div className="p-3 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)]">
      <div className="text-[10px] text-[var(--color-text-muted)] uppercase tracking-wide mb-1">{role}</div>
      <div className="text-sm font-mono font-semibold mb-1 truncate" title={model}>{model}</div>
      <div className="text-xs text-[var(--color-text-muted)]">
        temp {temperature} &middot; max {maxTokens} tok
      </div>
    </div>
  );
}
