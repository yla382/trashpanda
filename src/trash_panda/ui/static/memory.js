// The memory of every model: Disk, RAM with its pinned buffers, and the GPU, stacked from the top down. Where a block stands
// is its tier; its color and its tag say which model it belongs to. Everything drawn is something the code has (see
// docs/design/trace-viewer.md); the staged slots stand for tensors that wait on the GPU, nothing is allocated for them.
//
// Each band holds one group of blocks for each model. A model with many blocks is drawn with small tiles.

import { clear, el, place } from "./svg.js";

const MOVES = ["read", "load", "stage", "copy"];                 // the events that move a block from one place to another
const CX = 132, RIGHT = 654, GAP = 32;                           // the blocks start at CX, and nothing goes past RIGHT
const SLOT_H = 38;

const sizeOf = (m) => (m.blocks.length > 20
  ? { tw: 28, th: 20, pitch: 32, row: 26, compact: true }
  : { tw: 48, th: 26, pitch: 72, row: 46, compact: false });

export function buildMemory(svg, data, state) {
  clear(svg);
  const models = data.models;
  const byIndex = models.map((m, mi) => {
    const inst = data.instants.filter((e) => e.model === m.id);
    const replan = inst.find((e) => e.k === "replan");
    const demoted = replan ? replan.demoted || [] : [], promoted = replan ? replan.promoted || [] : [];
    const ids = (tier) => m.blocks.filter((b) => b.tier === tier).map((b) => b.id);
    const runsOf = {};
    m.events.filter((e) => e.k === m.visualRun).forEach((e) => { (runsOf[e.b] = runsOf[e.b] || []).push(e); });
    const freed = {};
    inst.filter((e) => e.k === "free").forEach((e) => { freed[e.b] = e.s; });
    return {
      m, mi, size: sizeOf(m), runsOf, freed, demoted, promoted, replanAt: replan ? replan.s : null,
      lists: { disk: m.blocks.map((b) => b.id), ram: [...ids("ram"), ...demoted], gpu: [...ids("gpu"), ...promoted] },
      early: data.pools.find((p) => p.kind === "early_loader" && p.models.includes(m.id)),
    };
  });

  // What the right of each band holds: the pinned pools, the places where a block stands while it is read into ordinary memory,
  // and the staged and running slots of the GPU.
  const pools = data.pools.map((p) => ({ ...p, n: Math.min(p.buffers, 6) }));
  const boxW = (p) => Math.max(16 + p.n * 64 + 8, 190);          // wide enough for the name of the pool
  const poolsW = pools.length ? Math.max(...pools.map(boxW)) : 0;
  const POOL_H = 76;
  const transitModels = byIndex.filter((x) => x.m.events.some((e) => e.via === "ram" && !(e.source === "chunked" && x.early)));
  const transitW = transitModels.length * 60;
  const ramRight = RIGHT - poolsW - (poolsW ? 12 : 0) - transitW - (transitW ? 12 : 0);
  const stageN = Math.max(1, Math.min(Math.max(...models.map((m) => m.plan.lookahead || 1)), 4));
  const runX = RIGHT - 88;
  const stageX = runX - 16 - stageN * 72;
  const room = { disk: RIGHT - CX, ram: ramRight - 12 - CX, gpu: stageX - 12 - CX };

  // The groups of blocks in each band, and the height of each band.
  const bands = {};
  for (const tier of ["disk", "ram", "gpu"]) {
    let y = 12;
    const groups = [];
    byIndex.forEach((x) => {
      const ids = x.lists[tier];
      if (!ids.length) return;
      const cols = Math.max(1, Math.min(ids.length, Math.floor(room[tier] / x.size.pitch)));
      const pitch = x.size.compact ? x.size.pitch : Math.min(72, Math.floor(room[tier] / cols));
      const rows = Math.ceil(ids.length / cols);
      groups.push({ x, ids, y, cols, pitch, rows });
      y += 18 + rows * x.size.row + 6;
    });
    const right = tier === "ram" ? pools.length * POOL_H + 8 : tier === "gpu" ? 84 : 0;
    bands[tier] = { groups, height: Math.max(y + 4, right, 64) };
  }
  const top = { disk: 10 };
  top.ram = top.disk + bands.disk.height + GAP;
  top.gpu = top.ram + bands.ram.height + GAP;
  svg.setAttribute("viewBox", "0 0 680 " + (top.gpu + bands.gpu.height + 14));

  const band = (tier, title, sub) => {
    el("rect", { class: "box", x: 16, y: top[tier], width: 648, height: bands[tier].height, rx: 8 }, svg);
    el("text", { class: "th", x: 28, y: top[tier] + 26 }, svg, title);
    el("text", { class: "muted", x: 28, y: top[tier] + 44 }, svg, sub);
  };
  const flow = (y, text) => {
    el("text", { class: "muted", x: CX - 9, y, "text-anchor": "end" }, svg, "↓");
    el("text", { class: "muted", x: CX, y }, svg, text);
  };
  band("disk", "Disk", "Checkpoint files");
  flow(top.disk + bands.disk.height + 20, "Read from the checkpoint file into RAM: ordinary memory, or a pinned buffer");
  band("ram", "RAM", "Compact form");
  flow(top.ram + bands.ram.height + 20, "Copied from RAM to the GPU, over PCIe");
  band("gpu", "GPU memory", "Kept on the GPU");

  // The slots of the blocks. A key is "model index:block".
  const slotPos = { disk: new Map(), ram: new Map(), gpu: new Map() };       // where a tile stands in a slot of each tier
  const slotRect = new Map();                                                // the outline of the slot of a kept block
  const fileName = (m) => ((m.checkpoint && m.checkpoint.path) || m.name).split(/[\\/]/).pop();
  for (const tier of ["disk", "ram", "gpu"]) {
    bands[tier].groups.forEach((g) => {
      const { x: info } = g, S = info.size;
      const caption = tier === "disk" ? fileName(info.m) : info.m.tag + "  " + info.m.name;
      el("text", { class: "muted", x: CX, y: top[tier] + g.y + 12 }, svg, caption);
      g.ids.forEach((id, i) => {
        const gx = CX + (i % g.cols) * g.pitch, gy = top[tier] + g.y + 18 + Math.floor(i / g.cols) * S.row;
        let pos;
        if (S.compact) {
          pos = [gx + (g.pitch - S.tw) / 2, gy + 2];
        } else {
          const r = el("rect", { class: "outline", x: gx, y: gy, width: g.pitch - 8, height: SLOT_H, rx: 4 }, svg);
          if (tier === "gpu") slotRect.set(info.mi + ":" + id, r);
          pos = [gx + (g.pitch - 8 - S.tw) / 2, gy + (SLOT_H - S.th) / 2];
        }
        slotPos[tier].set(info.mi + ":" + id, pos);
      });
    });
  }

  // The pinned pools, in the RAM band: each with its buffers.
  const pinned = new Map();                                                  // "pool|n" -> where a tile stands in that buffer
  pools.forEach((p, i) => {
    const x0 = RIGHT - poolsW, y0 = top.ram + 8 + i * POOL_H;
    el("rect", { class: "outline", x: x0, y: y0, width: poolsW, height: POOL_H - 8, rx: 6 }, svg);
    const label = p.kind === "early_loader" ? "Early loader of " + names(p.models) + " (own " + p.buffers + ")"
      : p.kind === "shared" ? "Pinned buffers (" + p.buffers + "), shared by " + names(p.models)
        : "Pinned buffers (" + p.buffers + "), " + names(p.models);
    const caption = el("text", { class: "muted", x: x0 + 8, y: y0 + 16 }, svg, label);
    fit(caption, poolsW - 16, label);                                          // a long name ends in "…"; the whole of it is the tooltip
    el("title", {}, caption, label);
    for (let n = 0; n < p.n; n++) {
      const bx = x0 + 8 + n * 64;
      el("rect", { class: "outline", x: bx, y: y0 + 24, width: 56, height: SLOT_H, rx: 4 }, svg);
      pinned.set(p.id + "|" + (n + 1), [bx + 4, y0 + 24 + 6]);
    }
  });
  function fit(textEl, width, full) {
    let n = full.length;
    while (n > 1 && textEl.getComputedTextLength() > width) {
      n -= 1;
      textEl.textContent = full.slice(0, n).trimEnd() + "…";
    }
  }
  function names(ids) {
    return ids.map((id) => (data.models.find((m) => m.id === id) || { tag: id }).tag).join(", ");
  }

  // Where a block stands while it is read into ordinary memory and copied to the GPU: in RAM, in no slot of its own.
  const transit = new Map();
  transitModels.forEach((x, k) => transit.set(x.mi, [ramRight + k * 60 + 8, top.ram + 8 + 18]));

  const slotY = top.gpu + 34;
  el("text", { class: "muted", x: stageX, y: slotY - 6 }, svg, "Staged ahead (" + stageN + ")");
  const staged = [];
  for (let n = 0; n < stageN; n++) {
    el("rect", { class: "outline", x: stageX + n * 72, y: slotY, width: 64, height: SLOT_H, rx: 4 }, svg);
    staged.push([stageX + n * 72, slotY]);
  }
  const runLabel = el("text", { class: "muted", x: runX, y: slotY - 6 }, svg, "Running");
  const runSlot = el("rect", { class: "outline", x: runX, y: slotY, width: 88, height: SLOT_H, rx: 4 }, svg);
  const centered = (corner, w, S) => [corner[0] + (w - S.tw) / 2, corner[1] + (SLOT_H - S.th) / 2];

  const group = el("g", {}, svg);
  const overlay = el("g", {}, svg);                                             // above the resting tiles: a tile on its way is never covered
  const tile = (x, id, moving) => {
    const S = x.size, b = x.m.byId.get(id);
    const g = el("g", { class: "tile " + x.m.color + (moving ? " mover" : "") }, moving ? overlay : group);
    el("title", {}, g, x.m.name + ": " + b.name + " (" + b.tier + ")");
    el("rect", { width: S.tw, height: S.th, rx: S.compact ? 3 : 4, "stroke-width": 0.5 }, g);
    const label = el("text", { x: S.tw / 2, y: S.th / 2, "text-anchor": "middle", "dominant-baseline": "central" }, g, S.compact ? String(id) : x.m.tag + id);
    if (S.compact) label.setAttribute("font-size", "11");
    g.bar = el("rect", { class: "bar", x: 0, y: S.th + 2, width: 0, height: 3, rx: 1.5 }, g);
    return g;
  };

  const byStart = (a, b) => a.s - b.s;
  const layers = byIndex.map((x) => {
    const { m, mi } = x;
    const key = (id) => mi + ":" + id;
    const tierOf = (id) => m.tierOf(id);
    const moves = m.events.filter((e) => MOVES.includes(e.k) && m.byId.has(e.b));
    const isLoad = (e) => e.k === "load" || tierOf(e.b) === "gpu" || e.to === "ram";   // the load puts a block where it is kept; the rest is a pass
    const spot = (name, id, cycle, seg) => {
      if (name === "gpu" || name === "staged") return tierOf(id) === "gpu" ? slotPos.gpu.get(key(id)) : centered(staged[cycle ? cycle.slot : 0], 64, x.size);
      if (name === "ram") return slotPos.ram.get(key(id)) || slotPos.disk.get(key(id));
      if (name === "disk") return slotPos.disk.get(key(id));
      const n = name.startsWith("pinned:") ? name.slice(7) : null;
      return (n && pinned.get(((seg && seg.pool) || m.pool) + "|" + n)) || slotPos.disk.get(key(id));
    };
    const disk = {}, kept = {}, alt = {}, mover = {}, loadSegs = {}, cycles = {};
    m.blocks.forEach((b) => {
      const id = b.id;
      disk[id] = tile(x, id);
      place(disk[id], slotPos.disk.get(key(id)));
      const mine = moves.filter((e) => e.b === id);
      const load = mine.filter(isLoad).sort(byStart);
      if (b.tier !== "disk") {
        kept[id] = tile(x, id);
        place(kept[id], slotPos[b.tier].get(key(id)));
        loadSegs[id] = load;
        if (load.length) kept[id].classList.add("hide");         // not there until the load puts it there
      }
      if (x.demoted.includes(id) || x.promoted.includes(id)) {   // a replan moved it to the other tier
        const tier = x.demoted.includes(id) ? "ram" : "gpu";
        alt[id] = tile(x, id);
        place(alt[id], slotPos[tier].get(key(id)));
        alt[id].classList.add("hide");
      }
      const pass = mine.filter((e) => !isLoad(e)).sort(byStart);
      if (pass.length) {
        mover[id] = tile(x, id, true);
        mover[id].classList.add("hide");
        cycles[id] = [];
        let cur = [];
        pass.forEach((s) => {                                    // a cycle: the moves of one block towards the GPU, and its run there
          cur.push(s);
          if (s.to === "gpu" || s.to === "staged") {
            const run = (x.runsOf[id] || []).find((r) => r.s >= s.e - 1e-6);
            cycles[id].push({ segs: cur, copyEnd: s.e, runS: run ? run.s : s.e, runE: run ? run.e : s.e, slot: 0 });
            cur = [];
          }
        });
        if (cur.length) cycles[id].push({ segs: cur, copyEnd: cur[cur.length - 1].e, runS: cur[cur.length - 1].e, runE: cur[cur.length - 1].e, slot: 0 });
      }
    });
    return { x, key, tierOf, spot, disk, kept, alt, mover, loadSegs, cycles };
  });

  // A block that arrives on the GPU before its turn waits in a staged slot, the lowest one that nobody holds then.
  const all = [];
  layers.forEach((L) => { for (const id in L.cycles) L.cycles[id].forEach((c) => all.push(c)); });
  all.sort((a, b) => a.segs[a.segs.length - 1].s - b.segs[b.segs.length - 1].s);
  const holds = staged.map(() => []);
  all.forEach((c) => {
    const from = c.segs[c.segs.length - 1].s;
    let k = holds.findIndex((list) => list.every((end) => end <= from + 1e-6));
    if (k < 0) k = all.indexOf(c) % staged.length;               // more staged blocks than slots: share one
    c.slot = k;
    holds[k].push(c.runS);
  });

  // A tile that is being loaded moves like the tiles of a pass: while it moves it is lifted into the layer above the others.
  const lift = (tile, up) => {
    if (tile.up === up) return;
    tile.up = up;
    tile.classList.toggle("mover", up);
    (up ? overlay : group).appendChild(tile);
  };

  const lerp = (a, b, q) => [a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q];
  const follow = (L, list, t, id, cycle) => {                    // the position along a chain of movements, null before it starts
    if (t < list[0].s) return null;
    let cur = list[0];
    list.forEach((s) => { if (s.s <= t) cur = s; });
    const from = L.spot(cur.from, id, cycle, cur), to = L.spot(cur.to, id, cycle, cur);
    if (t < cur.e) {
      const p = (t - cur.s) / Math.max(cur.e - cur.s, 1e-9);
      if (cur.via) {                                             // a load: the block is in RAM while it is read and copied
        const early = cur.source === "chunked" && L.x.early;     // the early loader copies through two pinned buffers of its own
        const via = early ? pinned.get(L.x.early.id + "|1") : transit.get(L.x.mi);
        if (via) {
          const edge = 0.12;
          return { pos: p < edge ? lerp(from, via, p / edge) : p > 1 - edge ? lerp(via, to, (p - 1 + edge) / edge) : via, bar: p, running: false };
        }
      }
      return { pos: lerp(from, to, p), bar: p, running: false };
    }
    return { pos: to, bar: null, running: false };
  };

  const stallOf = (t) => {
    for (const L of layers) {
      const e = L.x.m.events.find((ev) => ev.k === "wait" && ev.reason !== "pinned buffer" && ev.b !== undefined && t >= ev.s && t < ev.e);
      if (e) return { e, L };
    }
    return null;
  };

  return {
    update(t) {
      const stall = stallOf(t);
      layers.forEach((L) => {
        const { x } = L, m = x.m;
        const other = state.model && state.model !== m.id;
        m.blocks.forEach((b) => {
          const id = b.id;
          const runs = (x.runsOf[id] || []).some((r) => t >= r.s && t < r.e);
          const gone = x.freed[id] !== undefined && t >= x.freed[id];           // freed after it ran, in the final pass
          const replanned = x.replanAt !== null && t >= x.replanAt && (x.demoted.includes(id) || x.promoted.includes(id));
          let present = b.tier !== "disk" && !replanned;
          if (L.kept[id]) {
            const st = L.loadSegs[id].length ? follow(L, L.loadSegs[id], t, id, null) : { pos: slotPos[b.tier].get(L.key(id)), bar: null };
            present = !!st && !replanned;
            L.kept[id].classList.toggle("hide", !st || (replanned && x.demoted.includes(id)));
            if (st) {
              place(L.kept[id], st.pos);
              L.kept[id].bar.setAttribute("width", st.bar === null ? 0 : x.size.tw * st.bar);
            }
            lift(L.kept[id], !!st && st.bar !== null);
          }
          if (L.alt[id]) {
            L.alt[id].classList.toggle("hide", !replanned);
            if (x.promoted.includes(id) && L.kept[id]) L.kept[id].classList.toggle("dim", replanned);   // its RAM copy stays
          }
          let flying = null;
          if (L.mover[id]) {
            const cycle = L.cycles[id].find((c) => t >= c.segs[0].s && t < c.runE);
            if (cycle) {
              if (t < cycle.copyEnd) flying = follow(L, cycle.segs, t, id, cycle);
              else if (t < cycle.runS) flying = { pos: centered(staged[cycle.slot], 64, x.size), bar: null, running: false };   // waiting on the GPU for its turn
              else flying = { pos: centered([runX, slotY], 88, x.size), bar: null, running: true };
            }
            L.mover[id].classList.toggle("hide", !flying);
            L.mover[id].classList.toggle("run", !!flying && flying.running);
            if (flying) {
              place(L.mover[id], flying.pos);
              L.mover[id].bar.setAttribute("width", flying.bar === null ? 0 : x.size.tw * flying.bar);
            }
          }
          const home = L.kept[id] || L.disk[id];
          home.classList.toggle("run", runs && !flying && (b.tier === "disk" || present));     // a block that runs where it stands
          if (L.kept[id]) L.kept[id].classList.toggle("freed", gone);
          L.disk[id].classList.toggle("dim", (b.tier !== "disk" && present && !gone) || (b.tier === "disk" && !!flying));
          if (L.kept[id] && b.tier === "ram") L.kept[id].classList.toggle("dim", !!flying);
          [L.disk[id], L.kept[id], L.alt[id], L.mover[id]].forEach((tileEl) => { if (tileEl) tileEl.classList.toggle("other", !!other); });
        });
      });
      for (const [k, r] of slotRect) {
        const hit = stall && stall.L.x.mi + ":" + stall.e.b === k && stall.L.tierOf(stall.e.b) === "gpu";
        r.style.stroke = hit ? "var(--red-s)" : "";
        r.style.strokeWidth = hit ? "2" : "";
        r.style.strokeDasharray = hit ? "none" : "";
      }
      const streamStall = !!stall && stall.L.tierOf(stall.e.b) !== "gpu";
      runSlot.style.stroke = streamStall ? "var(--red-s)" : "";
      runSlot.style.strokeWidth = streamStall ? "2" : "";
      runSlot.style.strokeDasharray = streamStall ? "none" : "";
      runLabel.textContent = streamStall ? "waiting for " + stall.L.x.m.tag + stall.e.b : "Running";
      runLabel.style.fill = streamStall ? "var(--red-s)" : "";
    },
  };
}
