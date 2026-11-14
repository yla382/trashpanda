// Memory in use, for each model, worked out from the events: no recorder measures it yet, so these are estimates, and the page
// says so. A trace may carry counters (`vram`, `ram`, `pinned`); where it does, those are used and nothing is worked out.
//
//   VRAM    the weights outside the blocks, the kept blocks, the staged blocks (compact, waiting for their turn), and the
//           block that runs, expanded. Activations are not in the trace.
//   RAM     the blocks of the RAM tier that were read into memory (a mapped block is not counted: its pages are in the file cache).
//   pinned  the buffers of each pool: from `alloc` events if there are any, else the plan's estimate (each buffer is the
//           largest block rounded up to a power of two, as PyTorch does it).
//
// Nothing here touches the page, so it can be tried on its own.

const EPS = 1e-6;
const CHUNK = 64 * 2 ** 20;                                    // what the early loader copies through its pinned buffers at a time
const pow2 = (n) => (n > 0 ? 2 ** Math.ceil(Math.log2(n)) : 0);

export function buildUsage(data) {
  const intervals = { vram: [], ram: [], pinned: [] };
  const add = (resource, model, kind, s, e, bytes) => {
    if (bytes > 0 && e > s) intervals[resource].push({ model, kind, s, e, bytes });
  };

  data.models.forEach((m) => {
    const inst = data.instants.filter((x) => x.model === m.id);
    const freed = {};
    inst.filter((x) => x.k === "free").forEach((x) => { freed[x.b] = x.s; });
    const replan = inst.find((x) => x.k === "replan");
    const demoted = new Set(replan ? replan.demoted || [] : []), promoted = new Set(replan ? replan.promoted || [] : []);
    const runsOf = {};
    m.events.filter((e) => e.k === m.visualRun && e.b !== undefined).forEach((e) => { (runsOf[e.b] = runsOf[e.b] || []).push(e); });
    const stored = (id) => (m.byId.get(id) || {}).stored || 0;
    const expanded = (id) => (m.byId.get(id) || {}).expanded || 0;

    if (m.outside) add("vram", m.id, "outside", m.s, m.e, m.outside);
    m.blocks.forEach((b) => {
      const id = b.id, gone = freed[id] !== undefined ? freed[id] : m.e;
      const onGpu = (b.tier === "gpu" && !demoted.has(id)) || promoted.has(id);
      if (onGpu) {
        const loads = m.events.filter((e) => e.k === "load" && e.b === id);
        const from = promoted.has(id) ? replan.s : loads.length ? Math.max(...loads.map((e) => e.e)) : m.s;
        add("vram", m.id, "kept", from, gone, stored(id));
        if (expanded(id) && expanded(id) !== stored(id)) (runsOf[id] || []).forEach((r) => add("vram", m.id, "running", r.s, r.e, expanded(id)));
      } else {
        m.events.filter((e) => e.k === "copy" && e.b === id).forEach((c) => {     // a streamed block: staged, then running, each pass
          const run = (runsOf[id] || []).find((r) => r.s >= c.e - EPS);
          add("vram", m.id, "staged", c.e, run ? run.s : c.e, c.bytes || stored(id));
          if (run && expanded(id)) add("vram", m.id, "running", run.s, run.e, expanded(id));
        });
      }
      if (b.tier === "ram" && !promoted.has(id)) {
        const reads = m.events.filter((e) => e.k === "read" && e.b === id && e.to === "ram");
        if (reads.length) add("ram", m.id, "tier", Math.max(...reads.map((e) => e.e)), gone, stored(id));
      }
    });
  });

  data.pools.forEach((p) => {
    const owner = p.kind === "shared" ? "pool:" + p.id : p.models[0] || null;
    const users = data.models.filter((m) => p.models.includes(m.id));
    const mine = data.events.filter((e) => e.pool === p.id);
    const allocs = mine.filter((e) => e.k === "alloc");
    if (allocs.length) {                                       // allocations were recorded: a buffer holds its last size
      const end = Math.max(...users.map((m) => m.e), ...allocs.map((e) => e.e));
      const byBuffer = new Map();
      allocs.sort((a, b) => a.s - b.s).forEach((a) => { (byBuffer.get(a.buffer) || byBuffer.set(a.buffer, []).get(a.buffer)).push(a); });
      byBuffer.forEach((list) => list.forEach((a, i) => add("pinned", owner, "pool:" + p.id, a.e, i + 1 < list.length ? list[i + 1].s : end, a.bytes)));
      return;
    }
    if (p.kind === "early_loader") {                           // two buffers, 64 MB at a time, while the early loader copies
      const loads = data.events.filter((e) => e.k === "load" && e.source === "chunked" && p.models.includes(e.model));
      if (!loads.length) return;
      const biggest = Math.max(...loads.map((e) => e.bytes || 0), 0);
      add("pinned", owner, "pool:" + p.id, Math.min(...loads.map((e) => e.s)), Math.max(...loads.map((e) => e.e)),
        p.buffers * pow2(Math.min(CHUNK, biggest || CHUNK)));
      return;
    }
    if (!mine.length) return;                                  // a pool nobody used
    const largest = Math.max(...users.flatMap((m) => m.blocks.filter((b) => b.tier !== "gpu").map((b) => b.stored || 0)), 0);
    add("pinned", owner, "pool:" + p.id, Math.min(...mine.map((e) => e.s)), Math.max(...users.map((m) => m.e)), p.buffers * pow2(largest));
  });

  // Who the bytes belong to: each model, and each pinned pool that no one model owns.
  const owners = data.models.map((m) => ({ key: m.id, name: m.name, tag: m.tag, color: m.color, fill: "--" + m.color + "-f", stroke: "--" + m.color + "-s" }));
  data.pools.filter((p) => p.kind === "shared").forEach((p) => {
    owners.push({ key: "pool:" + p.id, name: "shared pinned pool", tag: "P", color: "c-gray", fill: "--gray-f", stroke: "--gray-s" });
  });

  const counters = data.counters || {};
  const at = (resource, t) => {
    const byModel = new Map();
    const put = (model, kind, bytes) => {
      const row = byModel.get(model) || byModel.set(model, { total: 0, kinds: {} }).get(model);
      row.total += bytes;
      row.kinds[kind] = (row.kinds[kind] || 0) + bytes;
    };
    const list = counters[resource];
    if (list && list.length) {                                 // measured: the last counter at or before t
      let last = null;
      list.forEach((c) => { if (c.t <= t + EPS) last = c; });
      if (last) for (const model in last.values) put(model, "measured", last.values[model]);
    } else {
      intervals[resource].forEach((x) => { if (t >= x.s && t < x.e) put(x.model, x.kind, x.bytes); });
    }
    let total = 0;
    byModel.forEach((r) => { total += r.total; });
    return { total, byModel };
  };

  return {
    at,
    owners,
    measured: (resource) => !!(counters[resource] && counters[resource].length),
    has: (resource) => !!(intervals[resource].length || (counters[resource] && counters[resource].length)),
    // The usage at n points between ws and we: for each model, its bytes at each point, and the total at each point.
    series(resource, ws, we, n = 160) {
      const times = [], totals = [];
      const per = new Map(owners.map((o) => [o.key, []]));
      for (let i = 0; i <= n; i++) {
        const t = ws + ((we - ws) * i) / n, now = at(resource, t);
        times.push(t);
        totals.push(now.total);
        per.forEach((values, model) => values.push((now.byModel.get(model) || { total: 0 }).total));
      }
      return { times, totals, per };
    },
    peak(resource) {
      let top = 0;
      for (let i = 0; i <= 400; i++) top = Math.max(top, at(resource, (data.duration * i) / 400).total);
      return top;
    },
  };
}
