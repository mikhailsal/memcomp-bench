import { useState, useMemo } from 'react';
import { useManifest } from '../lib/manifest';
import { formatCost, formatTokens } from '../lib/formatters';
import ConversationCard from '../components/ConversationCard';

type SortKey = 'date' | 'turns' | 'tokens' | 'cost' | 'profile';

export default function ConversationList() {
  const { manifest, loading, error } = useManifest();
  const [search, setSearch] = useState('');
  const [profileFilter, setProfileFilter] = useState('');
  const [sortBy, setSortBy] = useState<SortKey>('date');

  const filtered = useMemo(() => {
    if (!manifest) return [];
    let list = manifest.conversations;

    if (profileFilter) {
      list = list.filter(c => c.profile === profileFilter);
    }

    if (search) {
      const q = search.toLowerCase();
      list = list.filter(c =>
        c.profile.toLowerCase().includes(q) ||
        c.aiModel.toLowerCase().includes(q) ||
        c.humanModel.toLowerCase().includes(q) ||
        c.seedWords.some(w => w.toLowerCase().includes(q)) ||
        c.language.toLowerCase().includes(q)
      );
    }

    const sorted = [...list];
    switch (sortBy) {
      case 'date':
        sorted.sort((a, b) => b.startedAt.localeCompare(a.startedAt));
        break;
      case 'turns':
        sorted.sort((a, b) => b.turns - a.turns);
        break;
      case 'tokens':
        sorted.sort((a, b) => b.tokens - a.tokens);
        break;
      case 'cost':
        sorted.sort((a, b) => b.cost - a.cost);
        break;
      case 'profile':
        sorted.sort((a, b) => a.profile.localeCompare(b.profile));
        break;
    }

    return sorted;
  }, [manifest, search, profileFilter, sortBy]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-[var(--color-text-muted)]">Loading conversations...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <p className="text-red-400">Failed to load manifest: {error}</p>
      </div>
    );
  }

  if (!manifest) return null;

  return (
    <div className="animate-fade-in">
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-1">Conversations</h1>
        <p className="text-sm text-[var(--color-text-muted)]">
          {manifest.totalConversations} conversations &middot; {manifest.totalTurns.toLocaleString()} turns &middot; {formatTokens(manifest.conversations.reduce((s, c) => s + c.tokens, 0))} tokens &middot; {formatCost(manifest.totalCost)} total cost
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3 mb-6">
        <div className="relative flex-1">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--color-text-muted)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search by profile, model, seed, language..."
            className="w-full pl-10 pr-4 py-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-raised)] text-sm text-[var(--color-text)] placeholder:text-[var(--color-text-muted)] focus:outline-none focus:border-sky-500/50 transition-colors"
          />
        </div>

        <select
          value={profileFilter}
          onChange={e => setProfileFilter(e.target.value)}
          className="px-3 py-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-raised)] text-sm text-[var(--color-text)] focus:outline-none focus:border-sky-500/50 transition-colors"
        >
          <option value="">All profiles</option>
          {manifest.profiles.map(p => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>

        <select
          value={sortBy}
          onChange={e => setSortBy(e.target.value as SortKey)}
          className="px-3 py-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-raised)] text-sm text-[var(--color-text)] focus:outline-none focus:border-sky-500/50 transition-colors"
        >
          <option value="date">Newest first</option>
          <option value="turns">Most turns</option>
          <option value="tokens">Most tokens</option>
          <option value="cost">Highest cost</option>
          <option value="profile">By profile</option>
        </select>
      </div>

      {filtered.length === 0 ? (
        <div className="text-center py-12 text-[var(--color-text-muted)]">
          No conversations match your filters.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {filtered.map(c => (
            <ConversationCard key={c.id} conversation={c} />
          ))}
        </div>
      )}
    </div>
  );
}
