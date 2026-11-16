export function compilationRanges(events, stroke) {
  const merged = [];
  for (const e of events.filter(e => e.include && e.stroke === stroke).sort((a, b) => a.start - b.start)) {
    const last = merged.at(-1);
    if (last && e.start <= last[1]) last[1] = Math.max(last[1], e.end);
    else merged.push([e.start, e.end]);
  }
  return merged;
}

export function validateEvents(events, duration) {
  if (!Array.isArray(events) || events.length > 10000) throw new Error('Expected a list of up to 10,000 shots.');
  return events.map((e, index) => {
    if (![e.start, e.end].every(Number.isFinite) || e.start < 0 || e.start >= e.end || e.end > duration + .05) throw new Error(`Shot ${index + 1}: use start < end within the video.`);
    if (!['forehand', 'backhand', 'unknown', 'ignore'].includes(e.stroke) || typeof e.include !== 'boolean') throw new Error(`Shot ${index + 1}: invalid category or inclusion.`);
    return { ...e, id: index + 1, end: Math.min(e.end, duration) };
  });
}
