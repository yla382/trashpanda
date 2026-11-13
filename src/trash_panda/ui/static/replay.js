"use strict";

const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);

function el(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

function place(e, p) {
  e.style.transform = "translate(" + p[0] + "px," + p[1] + "px)";
}

const COLOR = { run: "c-teal", wait: "c-red", stage: "c-purple", copy: "c-purple", read: "c-gray" };
const STACK = ["read", "stage", "copy", "wait", "run"];       // the order of a block's time in the bars

fetch("/trace.json")
  .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
  .then(({ ok, d }) => {
    if (!ok) throw new Error(d.error);
    start(d);
  })
  .catch((e) => {
    $("error").textContent = e.message;
    $("error").hidden = false;
  });

function start(data) {
  const plan = data.trashPanda;
  const blocks = plan.blocks;
  const dur = plan.duration_us / 1000;                         // the run, in ms
  const events = data.traceEvents
    .filter((e) => e.ph === "X")
    .map((e) => ({ name: e.name, k: e.cat, b: e.args.block, s: e.ts / 1000, e: (e.ts + e.dur) / 1000, tid: e.tid, from: e.args.from, to: e.args.to }));
  const phases = (plan.phases || []).map((p) => ({ name: p.name, s: p.start_us / 1000, e: p.end_us / 1000 }));
  const threadName = {};
  data.traceEvents.filter((e) => e.ph === "M" && e.name === "thread_name").forEach((e) => { threadName[e.tid] = e.args.name; });
  const runsOf = {};
  events.filter((e) => e.k === "run").forEach((e) => { (runsOf[e.b] = runsOf[e.b] || []).push(e); });
  const tierOf = {};
  blocks.forEach((b) => { tierOf[b.id] = b.tier; });

  $("meta").textContent = blocks.length + " blocks, " + Math.round(dur) + " ms, " + plan.pinned_buffers + " pinned buffers" +
    (plan.note ? " (" + plan.note + ")" : "");
  $("view").hidden = false;

  const mem = buildMemory(blocks, plan.pinned_buffers, events, runsOf);
  const tl = buildTimeline(events, threadName, dur, phases);
  const bars = buildBars(blocks, events, phases);

  const scrub = $("scrub");
  scrub.max = dur;
  let t = 0, playing = false, last = 0;
  const speed = dur / 18000;                                   // the whole run plays in about 18 seconds

  function update() {
    scrub.value = t;
    const phase = phases.find((p) => t >= p.s && t < p.e);
    $("clock").textContent = "t = " + Math.round(t) + " ms" + (phase ? ", " + phase.name : "");
    mem.update(t);
    tl.update(t);
    bars.update(t);
    $("caption").textContent = describe(events, t, dur, tierOf);
  }

  function setButton() {
    $("play").textContent = playing ? "Pause" : "Play";
  }

  function frame(now) {
    if (!playing) return;
    t = Math.min(dur, t + (now - last) * speed);
    last = now;
    update();
    if (t >= dur) { playing = false; setButton(); return; }
    requestAnimationFrame(frame);
  }

  $("play").onclick = () => {
    if (playing) { playing = false; setButton(); return; }
    if (t >= dur) t = 0;
    playing = true;
    setButton();
    last = performance.now();
    requestAnimationFrame(frame);
  };
  scrub.oninput = () => { playing = false; setButton(); t = +scrub.value; update(); };
  $("tl").onclick = (ev) => {
    const r = $("tl").getBoundingClientRect();
    const x = ((ev.clientX - r.left) * tl.width) / r.width;
    if (x >= tl.x0) { playing = false; setButton(); t = Math.max(0, Math.min(dur, (x - tl.x0) / tl.k)); update(); }
  };
  setButton();
  update();
}

function pinnedName(where) {
  return where && where.startsWith("pinned:") ? "pinned buffer " + where.slice(7) : where;
}

function describe(events, t, dur, tierOf) {
  const order = { read: 0, stage: 1, copy: 2, wait: 3, run: 4 };
  const say = (e) => {
    const n = "block " + e.b;
    if (e.k === "run") return n + " runs";
    if (e.k === "wait") return "the GPU waits for " + n;
    if (e.k === "read") return e.to === "ram" ? n + " is loaded from disk into RAM" : n + " is read from disk into " + pinnedName(e.to);
    if (e.k === "stage") return n + " is copied from RAM into " + pinnedName(e.to);
    if (e.k === "copy") return n + " goes from " + pinnedName(e.from) + (tierOf[e.b] === "gpu" ? " into GPU memory" : " to the GPU");
    return e.k + " " + n;
  };
  const text = events.filter((e) => t >= e.s && t < e.e).sort((a, b) => order[b.k] - order[a.k]).map(say).join("; ");
  if (!text) return t >= dur ? "The run is done." : "Starting.";
  return text.charAt(0).toUpperCase() + text.slice(1) + ".";
}

function buildMemory(blocks, nbuf, events, runsOf) {
  const svg = $("mem");
  const TW = 48, TH = 26, SH = 38, ROW = 46, MINP = 62;                     // tile, slot height, distance between rows of slots, least distance between slots
  const CX = 132, RIGHT = 654;                                              // the slots start here, and may not go past RIGHT
  const tierOf = {};
  blocks.forEach((b) => { tierOf[b.id] = b.tier; });
  const lists = { disk: blocks.map((b) => b.id), ram: blocks.filter((b) => b.tier === "ram").map((b) => b.id),
                  gpu: blocks.filter((b) => b.tier === "gpu").map((b) => b.id) };         // every block is in the file; the other two are where it ends up

  const boxW = 16 + nbuf * 88 + 8;                                          // the pinned buffers, inside the RAM band, on its right
  const pinX = RIGHT - boxW;
  const slotX = 430;                                                        // the streaming slot, in the GPU band
  const room = { disk: RIGHT - CX, ram: pinX - 12 - CX, gpu: slotX - 12 - CX };
  const lay = {};
  for (const tier in lists) {
    const cols = Math.max(1, Math.min(lists[tier].length || 1, Math.floor(room[tier] / MINP)));
    const pitch = Math.min(72, Math.floor(room[tier] / cols));
    lay[tier] = { cols: cols, pitch: pitch, sw: pitch - 8, rows: Math.max(1, Math.ceil(lists[tier].length / cols)) };
  }
  const slotOff = { disk: 16, ram: 34, gpu: 34 };                           // from the top of a band to its first row of slots
  const height = {};
  for (const tier in lists) height[tier] = slotOff[tier] + lay[tier].rows * ROW + 4;
  const GAP = 32;
  const top = { disk: 10 };
  top.ram = top.disk + height.disk + GAP;
  top.gpu = top.ram + height.ram + GAP;
  svg.setAttribute("viewBox", "0 0 680 " + (top.gpu + height.gpu + 14));

  const band = (tier, title, sub) => {
    el("rect", { class: "box", x: 16, y: top[tier], width: 648, height: height[tier], rx: 8 }, svg);
    el("text", { class: "th", x: 28, y: top[tier] + 26 }, svg, title);
    el("text", { class: "muted", x: 28, y: top[tier] + 44 }, svg, sub);
  };
  const flow = (y, text) => {
    el("text", { class: "muted", x: CX - 9, y: y, "text-anchor": "end" }, svg, "↓");
    el("text", { class: "muted", x: CX, y: y }, svg, text);
  };
  band("disk", "Disk", "Checkpoint file");
  flow(top.disk + height.disk + 20, "Read from the checkpoint file, into RAM or into a pinned buffer");
  band("ram", "RAM", "Compact form");
  flow(top.ram + height.ram + 20, "Copied from a pinned buffer to the GPU, over PCIe");
  band("gpu", "GPU memory", "Kept on the GPU");

  const slotPos = { disk: {}, ram: {}, gpu: {} };                           // where a tile stands in a slot of each tier
  for (const tier in lists) {
    const L = lay[tier];
    lists[tier].forEach((id, i) => {
      const x = CX + (i % L.cols) * L.pitch, y = top[tier] + slotOff[tier] + Math.floor(i / L.cols) * ROW;
      el("rect", { class: "outline", x: x, y: y, width: L.sw, height: SH, rx: 4 }, svg);
      slotPos[tier][id] = [x + (L.sw - TW) / 2, y + (SH - TH) / 2];
    });
  }

  const pinY = top.ram + slotOff.ram;                                       // in line with the slots of the RAM blocks
  el("rect", { class: "outline", x: pinX, y: top.ram + 8, width: boxW, height: slotOff.ram - 8 + SH + 8, rx: 6 }, svg);
  el("text", { class: "muted", x: pinX + 12, y: top.ram + 24 }, svg, "Pinned buffers (" + nbuf + ")");
  const spots = {};
  for (let n = 0; n < nbuf; n++) {
    const x = pinX + 16 + n * 88;
    el("rect", { class: "outline", x: x, y: pinY, width: 64, height: SH, rx: 4 }, svg);
    spots["pinned:" + (n + 1)] = [x + (64 - TW) / 2, pinY + (SH - TH) / 2];
  }

  const slotY = top.gpu + slotOff.gpu;
  el("text", { class: "muted", x: slotX, y: slotY - 6 }, svg, "Streaming slot, expanded to bf16");
  const slot = el("rect", { class: "outline", x: slotX, y: slotY, width: 88, height: SH, rx: 4 }, svg);
  const waitText = el("text", { x: slotX + 104, y: slotY + SH / 2, "dominant-baseline": "central", opacity: 0 }, svg, "");
  waitText.style.fill = "var(--red-s)";
  spots.S = [slotX + (88 - TW) / 2, slotY + (SH - TH) / 2];

  const group = el("g", {}, svg);
  const tile = (id, cls) => {
    const b = blocks.find((x) => x.id === id);
    const g = el("g", { class: "tile " + cls }, group);
    el("title", {}, g, b.name + " (" + b.tier + ")");
    el("rect", { width: TW, height: TH, rx: 4, "stroke-width": 0.5 }, g);
    el("text", { x: TW / 2, y: TH / 2, "text-anchor": "middle", "dominant-baseline": "central" }, g, "b" + id);
    g.bar = el("rect", { class: "bar", x: 0, y: TH + 2, width: 0, height: 3, rx: 1.5 }, g);
    return g;
  };
  const tierClass = { gpu: "c-teal", ram: "c-purple", disk: "c-gray" };

  // Where a movement starts or ends. "gpu" is the block's own slot for a block that is kept on the GPU, and the streaming slot for one that is not.
  const spot = (name, id) => {
    if (name === "gpu") return tierOf[id] === "gpu" ? slotPos.gpu[id] : spots.S;
    if (name === "ram") return slotPos.ram[id] || slotPos.disk[id];
    if (name === "disk") return slotPos.disk[id];
    return spots[name] || slotPos.disk[id];
  };

  const moves = events.filter((e) => e.k === "read" || e.k === "stage" || e.k === "copy");
  const isLoad = (e) => tierOf[e.b] === "gpu" || e.to === "ram";            // the load puts a block where it is kept; the rest is a pass
  const byStart = (a, b) => a.s - b.s;
  const diskTile = {}, kept = {}, mover = {}, loadSegs = {}, cycles = {};
  blocks.forEach((b) => {
    const id = b.id;
    diskTile[id] = tile(id, "c-gray");
    place(diskTile[id], slotPos.disk[id]);
    const mine = moves.filter((e) => e.b === id);
    const load = mine.filter(isLoad).sort(byStart);
    if (b.tier !== "disk") {
      kept[id] = tile(id, tierClass[b.tier]);
      place(kept[id], slotPos[b.tier][id]);
      loadSegs[id] = load;
      if (load.length) kept[id].classList.add("hide");                      // not there until the load puts it there
    }
    const pass = mine.filter((e) => !isLoad(e)).sort(byStart);
    if (pass.length) {
      mover[id] = tile(id, tierClass[b.tier]);
      mover[id].classList.add("hide");
      cycles[id] = [];
      let cur = [];
      pass.forEach((s) => {                                                 // a cycle ends when the block has run on the GPU
        cur.push(s);
        if (s.to === "gpu") {
          const run = (runsOf[id] || []).find((r) => r.s >= s.e - 1e-6);
          cycles[id].push({ segs: cur, end: run ? run.e : s.e });
          cur = [];
        }
      });
      if (cur.length) cycles[id].push({ segs: cur, end: cur[cur.length - 1].e });
    }
  });

  const follow = (list, t, id) => {                                         // the position along a chain of movements, null before it starts
    if (t < list[0].s) return null;
    let cur = list[0];
    list.forEach((s) => { if (s.s <= t) cur = s; });
    const from = spot(cur.from, id), to = spot(cur.to, id);
    if (t < cur.e) {
      const p = (t - cur.s) / (cur.e - cur.s);
      return { pos: [from[0] + (to[0] - from[0]) * p, from[1] + (to[1] - from[1]) * p], bar: p, atSlot: false };
    }
    return { pos: to, bar: null, atSlot: cur.to === "gpu" && tierOf[id] !== "gpu" };
  };

  return {
    update(t) {
      blocks.forEach((b) => {
        const id = b.id;
        const running = (runsOf[id] || []).some((r) => t >= r.s && t < r.e);
        let keptShown = b.tier !== "disk";
        if (kept[id]) {
          const st = loadSegs[id].length ? follow(loadSegs[id], t, id) : { pos: slotPos[b.tier][id], bar: null };
          keptShown = !!st;
          kept[id].classList.toggle("hide", !st);
          if (st) {
            place(kept[id], st.pos);
            kept[id].bar.setAttribute("width", st.bar === null ? 0 : TW * st.bar);
          }
        }
        let flying = null;
        if (mover[id]) {
          const cycle = cycles[id].find((c) => t >= c.segs[0].s && t < c.end);
          flying = cycle ? follow(cycle.segs, t, id) : null;
          mover[id].classList.toggle("hide", !flying);
          mover[id].classList.toggle("run", !!flying && flying.atSlot && running);
          if (flying) {
            place(mover[id], flying.pos);
            mover[id].bar.setAttribute("width", flying.bar === null ? 0 : TW * flying.bar);
          }
        }
        if (kept[id] && b.tier === "gpu") kept[id].classList.toggle("run", keptShown && running);
        diskTile[id].classList.toggle("dim", (b.tier !== "disk" && keptShown) || (b.tier === "disk" && !!flying));
        if (kept[id] && b.tier === "ram") kept[id].classList.toggle("dim", !!flying);
      });
      const waiting = events.find((e) => e.k === "wait" && t >= e.s && t < e.e);
      slot.style.stroke = waiting ? "var(--red-s)" : "";
      slot.style.strokeWidth = waiting ? "2" : "";
      slot.style.strokeDasharray = waiting ? "none" : "";
      waitText.setAttribute("opacity", waiting ? 1 : 0);
      waitText.textContent = waiting ? "waiting for b" + waiting.b : "";
    },
  };
}

function buildTimeline(events, threadName, dur, phases) {
  const svg = $("tl");
  const X0 = 196, W = 444;                                     // the track names need the room on the left
  const steps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
  const step = steps.find((s) => dur / s <= 8) || 10000;
  const axis = Math.ceil(dur / step) * step;
  const K = W / axis;
  const tids = [...new Set(events.map((e) => e.tid))].sort((a, b) => a - b);
  const rowY = (i) => 44 + i * 30;
  const height = rowY(tids.length) + 36;
  svg.setAttribute("viewBox", "0 0 680 " + height);
  for (let m = 0; m <= axis; m += step) {
    const x = X0 + m * K;
    el("text", { class: "muted", x: x, y: 10, "text-anchor": "middle" }, svg, m === 0 ? "0 ms" : String(m));
    el("line", { x1: x, y1: 14, x2: x, y2: 20, stroke: "var(--line-strong)", "stroke-width": 0.5 }, svg);
  }
  phases.forEach((p, i) => {
    el("text", { class: "muted", x: X0 + p.s * K + 4, y: 34 }, svg, p.name);
    if (i > 0) el("line", { x1: X0 + p.s * K, y1: 24, x2: X0 + p.s * K, y2: rowY(tids.length) - 4, stroke: "var(--line-strong)", "stroke-width": 0.8, "stroke-dasharray": "4 3" }, svg);
  });
  tids.forEach((tid, i) => {
    el("rect", { class: "box", x: X0, y: rowY(i), width: W, height: 24, rx: 4 }, svg);
    el("text", { class: "muted", x: 16, y: rowY(i) + 12, "dominant-baseline": "central" }, svg, threadName[tid] || "track " + tid);
  });
  const slices = events.map((e) => {
    const w = Math.max(1, (e.e - e.s) * K);
    const y = rowY(tids.indexOf(e.tid));
    const g = el("g", { class: COLOR[e.k] || "c-gray" }, svg);
    el("title", {}, g, e.name + ": " + Math.round(e.e - e.s) + " ms, from " + Math.round(e.s) + " ms");      // shown on hover, also where the slice is too small for a label
    el("rect", { x: X0 + e.s * K, y: y + 2, width: w, height: 20, rx: 3, "stroke-width": 0.5 }, g);
    if (e.k !== "wait" && w >= 18) el("text", { x: X0 + e.s * K + w / 2, y: y + 12, "text-anchor": "middle", "dominant-baseline": "central" }, g, "b" + e.b);
    return { g: g, s: e.s };
  });
  const head = el("line", { x1: X0, y1: 24, x2: X0, y2: rowY(tids.length) - 4, stroke: "var(--amber)", "stroke-width": 2 }, svg);
  const legend = [["c-teal", "Block runs", "run"], ["c-purple", "Copies", "copy"], ["c-gray", "Disk reads", "read"], ["c-red", "Waiting", "wait"]]
    .filter((l) => events.some((e) => e.k === l[2] || (l[2] === "copy" && e.k === "stage")));
  let lx = X0;
  legend.forEach((l) => {
    el("rect", { class: l[0], x: lx, y: rowY(tids.length) + 6, width: 12, height: 12, rx: 2, "stroke-width": 0.5 }, svg);
    el("text", { x: lx + 18, y: rowY(tids.length) + 12, "dominant-baseline": "central" }, svg, l[1]);
    lx += 18 + l[1].length * 7 + 18;
  });
  return {
    x0: X0, k: K, width: 680,
    update(t) {
      const x = X0 + t * K;
      head.setAttribute("x1", x);
      head.setAttribute("x2", x);
      slices.forEach((s) => { s.g.style.opacity = t >= s.s ? 1 : 0.3; });
    },
  };
}

function buildBars(blocks, events, phases) {
  const svg = $("bars");
  const rowY = (i) => 28 + i * 22;
  const sums = blocks.map((b) => {
    const mine = events.filter((e) => e.b === b.id);
    return { b: b, parts: STACK.map((k) => mine.filter((e) => e.k === k).map((e) => ({ k: k, ms: e.e - e.s }))).flat() };
  });
  const totals = sums.map((s) => s.parts.reduce((a, p) => a + p.ms, 0));
  const scale = 250 / Math.max(1, ...totals);
  svg.setAttribute("viewBox", "0 0 680 " + Math.max(rowY(blocks.length) + 16, 204));
  el("text", { class: "muted", x: 16, y: 12 }, svg, "Where each block's time went");
  const hl = el("rect", { x: 8, y: 0, width: 384, height: 20, rx: 4, opacity: 0 }, svg);
  sums.forEach((s, i) => {
    el("text", { x: 16, y: rowY(i) + 8, "dominant-baseline": "central" }, svg, "b" + s.b.id);
    let x = 50;
    s.parts.forEach((p) => {
      el("rect", { class: COLOR[p.k], x: x, y: rowY(i), width: p.ms * scale, height: 16, "stroke-width": 0.5 }, svg);
      x += p.ms * scale;
    });
    el("text", { class: "muted", x: x + 6, y: rowY(i) + 8, "dominant-baseline": "central" }, svg, Math.round(totals[i]) + " ms");
  });
  el("rect", { class: "box", x: 400, y: 24, width: 264, height: 176, rx: 8 }, svg);
  const line = (label, y) => {
    el("text", { class: "muted", x: 416, y: y }, svg, label);
    return el("text", { class: "th", x: 648, y: y, "text-anchor": "end" }, svg, "0 ms");
  };
  const tElapsed = line("Elapsed", 48), tLoad = line("Loading so far", 72), tRun = line("Computing so far", 96), tWait = line("Waiting so far", 120);
  el("line", { x1: 416, y1: 134, x2: 648, y2: 134, stroke: "var(--line)", "stroke-width": 0.5 }, svg);
  const tSum = line("Loading + run + wait", 156);
  const tCheck = el("text", { class: "muted", x: 416, y: 182 }, svg, "");
  const overlap = (list, t) => list.reduce((a, e) => a + Math.max(0, Math.min(t, e.e) - e.s), 0);
  const ofKind = (k) => events.filter((e) => e.k === k);
  const loads = phases.filter((p) => p.name === "load");
  return {
    update(t) {
      const load = overlap(loads, t), run = overlap(ofKind("run"), t), wait = overlap(ofKind("wait"), t);
      tElapsed.textContent = Math.round(t) + " ms";
      tLoad.textContent = Math.round(load) + " ms";
      tRun.textContent = Math.round(run) + " ms";
      tWait.textContent = Math.round(wait) + " ms";
      tSum.textContent = Math.round(load + run + wait) + " ms";
      tCheck.textContent = Math.abs(load + run + wait - t) < 1 ? "Matches elapsed: all time is explained" : "Does not match elapsed: events are missing";
      const cur = events.find((e) => (e.k === "run" || e.k === "wait") && t >= e.s && t < e.e);
      if (cur) {
        hl.setAttribute("y", rowY(blocks.findIndex((b) => b.id === cur.b)) - 2);
        hl.setAttribute("opacity", 0.18);
        hl.style.fill = cur.k === "wait" ? "var(--red-s)" : "var(--amber)";
      } else {
        hl.setAttribute("opacity", 0);
      }
    },
  };
}
