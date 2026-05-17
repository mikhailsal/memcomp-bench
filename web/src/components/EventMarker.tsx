import type { ConversationEvent } from '../lib/types';

interface EventMarkerProps {
  event: ConversationEvent;
}

export default function EventMarker({ event: e }: EventMarkerProps) {
  const parts: string[] = [];
  if (e.current_topic) parts.push(`topic: ${e.current_topic}`);
  if (e.topic_changed !== null) parts.push(e.topic_changed ? 'changed' : 'unchanged');
  if (e.nudge_injected !== null) parts.push(e.nudge_injected ? 'nudge sent' : 'no nudge');
  if (e.suppression_reason) parts.push(`suppressed: ${e.suppression_reason}`);

  return (
    <div className="flex items-center gap-3 my-3">
      <div className="flex-1 h-px bg-gradient-to-r from-transparent via-amber-500/30 to-transparent" />
      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-amber-500/20 bg-amber-500/5">
        <span className="text-xs">{'\uD83D\uDD14'}</span>
        <span className="text-[10px] font-semibold text-amber-400 uppercase tracking-wide">
          {e.event_type}
        </span>
        <span className="text-[10px] text-amber-400/60">
          turn {e.turn_number} &middot; {e.source}
        </span>
      </div>
      <div className="flex-1 h-px bg-gradient-to-r from-transparent via-amber-500/30 to-transparent" />

      {parts.length > 0 && (
        <div className="absolute hidden group-hover:block" />
      )}
    </div>
  );
}
