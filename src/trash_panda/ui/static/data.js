// The trace as the page uses it: formats 1 and 2 both become the same shape.
//
//   { duration, note, models, tracks, pools, phases, events, instants }
//
// Times are in milliseconds. An event is { name, k (the category), model, b (the block), s, e, tid, from, to, via, reason,
// pass, source, pool, bytes }. See docs/design/trace-format.md for the format.

const COLORS = 5;                                  // m0 to m4 in style.css: one color for each model, used again after that

export function fmtBytes(n) {
  if (n === null || n === undefined) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let x = n, i = 0;
  while (x >= 1000 && i < units.length - 1) { x /= 1000; i++; }
  return (i === 0 ? String(x) : x.toFixed(x < 10 ? 2 : 1)) + " " + units[i];
}

export function fmtMs(ms) {
  return ms >= 10000 ? (ms / 1000).toFixed(1) + " s" : ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms";
}

export function normalize(raw) {
  const plan = raw.trashPanda;
  const data = (plan.version || 1) >= 2 ? fromV2(raw, plan) : fromV1(raw, plan);
  data.phases.sort((a, b) => a.s - b.s);
  data.events.sort((a, b) => a.s - b.s);
  data.models.forEach((m, i) => finish(m, i, data));
  const tags = new Set();
  data.models.forEach((m) => {                      // a short tag for each model, different for each
    const base = (m.tag || m.name || m.id).replace(/[^A-Za-z0-9]/g, "").slice(0, 1).toUpperCase() || "M";
    let tag = m.tag || base, n = 1;
    while (tags.has(tag)) tag = base + ++n;
    tags.add(tag);
    m.tag = tag;
  });
  return data;
}

function event(e, a) {
  return { name: e.name, k: e.cat, model: a.model, b: a.block, s: e.ts / 1000, e: (e.ts + (e.dur || 0)) / 1000, tid: e.tid,
           from: a.from, to: a.to, via: a.via, reason: a.reason, pass: a.pass, source: a.source, pool: a.pool,
           bytes: a.bytes, buffer: a.buffer, tier: a.tier, demoted: a.demoted, promoted: a.promoted };
}

function fromV2(raw, plan) {
  const models = plan.models.map((m) => ({
    id: m.id, name: m.name, tag: m.tag, startedBy: m.started_by, checkpoint: m.checkpoint || null, dtypes: m.dtypes || null,
    blocks: m.blocks.map((b) => ({ id: b.id, name: b.name, tier: b.tier, stored: b.stored_bytes, expanded: b.expanded_bytes })),
    outside: m.outside_bytes, plan: m.plan || {}, options: m.options || {}, pool: m.pool || null,
  }));
  const events = [], instants = [], counters = {};
  raw.traceEvents.forEach((e) => {
    if (e.ph === "X") events.push(event(e, e.args || {}));
    else if (e.ph === "i") instants.push(event(e, e.args || {}));
    else if (e.ph === "C") (counters[e.name] = counters[e.name] || []).push({ t: e.ts / 1000, values: e.args || {} });
  });
  for (const name in counters) counters[name].sort((a, b) => a.t - b.t);
  return {
    duration: plan.duration_us / 1000, note: plan.note || "", models, events, instants, counters,
    tracks: plan.tracks.map((t) => ({ id: t.id, name: t.name, kind: t.kind, role: t.role, model: t.model || null })),
    pools: (plan.pools || []).map((p) => ({ id: p.id, kind: p.kind, buffers: p.buffers, models: p.models || [] })),
    phases: (plan.phases || []).map((p) => ({ model: p.model, name: p.name, s: p.start_us / 1000, e: p.end_us / 1000 })),
  };
}

function fromV1(raw, plan) {
  const id = "model";
  const names = {};
  raw.traceEvents.filter((e) => e.ph === "M" && e.name === "thread_name").forEach((e) => { names[e.tid] = e.args.name; });
  const events = [];
  raw.traceEvents.forEach((e) => {
    if (e.ph === "X") events.push(event(e, { ...e.args, model: id }));
  });
  const tids = [...new Set(events.map((e) => e.tid))].sort((a, b) => a - b);
  const role = (n) => (/main/i.test(n) ? "main" : /worker/i.test(n) ? "worker" : /load/i.test(n) ? "loader" : /stream/i.test(n) ? "copy_stream" : "worker");
  return {
    duration: plan.duration_us / 1000, note: plan.note || "", instants: [], counters: {},
    models: [{
      id, name: "model", startedBy: "stage", checkpoint: null, dtypes: null,
      blocks: plan.blocks.map((b) => ({ id: b.id, name: b.name, tier: b.tier, stored: plan.block_bytes || null })), outside: null,
      plan: { lookahead: plan.lookahead || 1 },
      options: { overlap: plan.overlap, ram_in_memory: plan.ram_in_memory, pinned_buffers: plan.pinned_buffers }, pool: "pool",
    }],
    tracks: tids.map((t) => ({ id: t, name: names[t] || "track " + t, kind: /stream/i.test(names[t] || "") ? "gpu" : "cpu",
                              role: role(names[t] || ""), model: null })),
    pools: [{ id: "pool", kind: "stage", buffers: plan.pinned_buffers, models: [id] }],
    phases: (plan.phases || []).map((p) => ({ model: id, name: p.name, s: p.start_us / 1000, e: p.end_us / 1000 })),
    events,
  };
}

// What a model needs besides its own fields: its color, its blocks by number, its events, and its passes.
function finish(m, index, data) {
  m.color = "m" + (index % COLORS);
  m.byId = new Map(m.blocks.map((b) => [b.id, b]));
  m.events = data.events.filter((e) => e.model === m.id);
  m.phases = data.phases.filter((p) => p.model === m.id);
  m.passes = m.phases.filter((p) => /^pass/i.test(p.name));
  m.tiers = { gpu: 0, ram: 0, disk: 0 };
  m.blocks.forEach((b) => { m.tiers[b.tier]++; });
  m.visualRun = m.events.some((e) => e.k === "run") ? "run" : "call";     // what shows a block running
  m.workRun = m.events.some((e) => e.k === "call") ? "call" : "run";      // what the main thread's totals add up
  const times = m.phases.length ? m.phases : m.events;
  m.s = times.length ? Math.min(...times.map((x) => x.s)) : 0;
  m.e = times.length ? Math.max(...times.map((x) => x.e)) : data.duration;
  m.tierOf = (id) => (m.byId.get(id) || {}).tier;
}

export const STARTED = { stage: "stage()", preload: "preload(), then stage()", coordinator: "a Coordinator step" };
