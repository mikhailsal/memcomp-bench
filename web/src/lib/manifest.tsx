import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';
import type { Manifest, ConversationSummary } from './types';

interface ManifestState {
  manifest: Manifest | null;
  loading: boolean;
  error: string | null;
}

const ManifestContext = createContext<ManifestState>({
  manifest: null,
  loading: true,
  error: null,
});

export function ManifestProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ManifestState>({
    manifest: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    const base = import.meta.env.BASE_URL;
    fetch(`${base}manifest.json`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: Manifest) => setState({ manifest: data, loading: false, error: null }))
      .catch(e => setState({ manifest: null, loading: false, error: e.message }));
  }, []);

  return (
    <ManifestContext.Provider value={state}>
      {children}
    </ManifestContext.Provider>
  );
}

export function useManifest() {
  return useContext(ManifestContext);
}

export function useConversation(id: string | undefined): ConversationSummary | undefined {
  const { manifest } = useManifest();
  if (!manifest || !id) return undefined;
  return manifest.conversations.find(c => c.id === id);
}
