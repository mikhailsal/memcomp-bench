export default function About() {
  return (
    <div className="animate-fade-in max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">About memcomp-bench</h1>

      <div className="space-y-6 text-sm leading-relaxed text-[var(--color-text-muted)]">
        <div className="p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)]">
          <h2 className="text-lg font-semibold text-[var(--color-text)] mb-2">Memory Compression Benchmark</h2>
          <p>
            memcomp-bench measures personality preservation through context compression by generating long
            multi-turn conversations between two LLM models via the OpenRouter API.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)]">
          <h2 className="text-lg font-semibold text-[var(--color-text)] mb-2">How It Works</h2>
          <p className="mb-3">
            One model acts as an <strong className="text-emerald-400">AI companion</strong> with an independent
            personality, tool-call-based communication, and a randomly generated personality seed. A second model
            simulates a <strong className="text-blue-400">human</strong> with a detailed backstory and a
            pre-generated conversation plan.
          </p>
          <p>
            A lightweight topic judge model periodically checks for topic staleness and injects nudges to keep
            the dialogue natural and flowing.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)]">
          <h2 className="text-lg font-semibold text-[var(--color-text)] mb-2">Communication Protocol</h2>
          <p>
            The AI communicates with the human exclusively via{' '}
            <code className="bg-black/30 px-1 py-0.5 rounded text-purple-400 font-mono text-xs">write_message_to_human</code>{' '}
            tool calls. This indirect channel allows the AI to maintain an inner monologue (reasoning field in the
            tool call) separate from its visible message, providing a window into the model&apos;s thinking process.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)]">
          <h2 className="text-lg font-semibold text-[var(--color-text)] mb-2">Human Profiles</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-3">
            {PROFILES.map(p => (
              <div key={p.name} className="flex items-start gap-2 p-2 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)]">
                <span className="text-sm shrink-0">{p.emoji}</span>
                <div>
                  <div className="text-sm font-semibold text-[var(--color-text)]">{p.name}</div>
                  <div className="text-xs text-[var(--color-text-muted)]">{p.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)]">
          <h2 className="text-lg font-semibold text-[var(--color-text)] mb-2">What You See Here</h2>
          <p>
            This viewer displays the generated conversations. Each conversation shows the full dialogue with
            the AI&apos;s inner monologue, native reasoning, and response drafts visible in collapsible panels.
            System events like topic judge nudges appear as inline markers between turns.
          </p>
        </div>

        <div className="text-center text-xs text-[var(--color-text-muted)] py-4">
          <a
            href="https://github.com/mikhailsal/memcomp-bench"
            target="_blank"
            rel="noopener"
            className="text-sky-400 hover:underline"
          >
            View on GitHub
          </a>
        </div>
      </div>
    </div>
  );
}

const PROFILES = [
  { name: 'Marcus', emoji: '\uD83C\uDFB7', desc: 'Software architect, philosopher, jazz guitarist from Portland' },
  { name: 'Anya', emoji: '\uD83C\uDFA8', desc: 'Illustrator and art teacher from Berlin, originally Moscow' },
  { name: 'James', emoji: '\uD83D\uDCDA', desc: 'History teacher from Chicago, opinionated and witty' },
  { name: 'Priya', emoji: '\uD83D\uDD2D', desc: 'Biotech researcher, amateur astronomer from Bangalore/SF' },
  { name: 'Leo', emoji: '\uD83D\uDCF0', desc: 'Independent journalist from London, digital rights advocate' },
  { name: 'Michael', emoji: '\uD83D\uDCBB', desc: 'AI entrepreneur from Tel Aviv, technically demanding' },
  { name: 'Nathan', emoji: '\uD83E\uDDE0', desc: 'Reclusive former programmer, runs AI experiments obsessively' },
  { name: 'Vitaly', emoji: '\uD83D\uDE12', desc: 'Burned-out programmer from Minsk, cynical and darkly humorous' },
  { name: 'Alex', emoji: '\uD83E\uDD16', desc: 'Special: an AI posing as a human, with a revelation arc' },
];
