// The timeline: a track for each thread and each CUDA stream, the CPU's first, then the GPU's, which run on the GPU's own
// clock. A slice for each event, in the color of its model; a red slice is a wait. Where a compute stream has gaps between
// the blocks of a pass, the gap is shown as GPU idle (worked out from the runs, not recorded).
//
// The wheel zooms, dragging pans, a click moves the playhead, and the pointer over a slice says what it is.

import { clear, el, tickStep } from "./svg.js";
import { fmtBytes, fmtMs } from "./data.js";

const X0 = 196, W = 444, HEADER = 18;
const IDLE_MIN = 0.5;                                          // a gap of the GPU shorter than this (ms) is not called idle

export function buildTimeline(svg, data, state, hooks = {}) {
  const used = data.tracks.filter((t) => data.events.some((e) => e.tid === t.id));
  const groups = [["cpu", "CPU threads"], ["gpu", "GPU streams, on the GPU's own clock"]]
    .map(([kind, label]) => ({ label, tracks: used.filter((t) => t.kind === kind).sort((a, b) => a.id - b.id) }))
    .filter((g) => g.tracks.length);
  const rowY = new Map(), headers = [];
  let y = 44;
  groups.forEach((g) => {
    headers.push({ label: g.label, y });
    y += HEADER;
    g.tracks.forEach((t) => { rowY.set(t.id, y); y += 30; });
  });
  const bottom = y;
  svg.setAttribute("viewBox", "0 0 680 " + (bottom + 36));
  const models = new Map(data.models.map((m) => [m.id, m]));
  const trackOf = new Map(data.tracks.map((t) => [t.id, t]));
  const dur = data.duration;
  const minSpan = Math.max(dur / 5000, 0.05);
  const view = { ws: 0, we: dur, k: W / dur, slices: [], head: null };
  const xOf = (t) => X0 + (t - view.ws) * view.k;
  const px = (ev) => {                                         // a pointer's x, in the units of the drawing
    const r = svg.getBoundingClientRect();
    return ((ev.clientX - r.left) * 680) / r.width;
  };

  // The gaps between the runs of a compute stream, in one pass of one model.
  const idle = [];
  used.filter((t) => t.role === "compute_stream").forEach((t) => {
    const runs = data.events.filter((e) => e.tid === t.id && e.k === "run").sort((a, b) => a.s - b.s);
    runs.forEach((a, i) => {
      const b = runs[i + 1];
      if (b && a.model === b.model && a.pass === b.pass && b.s - a.e >= IDLE_MIN) idle.push({ tid: t.id, s: a.e, e: b.s, model: a.model, before: a.b, after: b.b });
    });
  });

  function draw() {
    clear(svg);
    view.slices = [];
    const defs = el("defs", {}, svg);
    const hatch = el("pattern", { id: "hatch", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
    el("rect", { width: 6, height: 6, fill: "var(--red-f)" }, hatch);
    el("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: "var(--red-s)", "stroke-width": 2 }, hatch);
    const span = view.we - view.ws;
    const step = tickStep(span);
    for (let m = Math.ceil(view.ws / step) * step; m <= view.we + 1e-6; m += step) {
      el("text", { class: "muted", x: xOf(m), y: 10, "text-anchor": "middle" }, svg, m === 0 ? "0 ms" : String(+m.toFixed(3)));
      el("line", { x1: xOf(m), y1: 14, x2: xOf(m), y2: 20, stroke: "var(--line-strong)", "stroke-width": 0.5 }, svg);
    }
    const selected = models.get(state.model);
    let labelEnd = 0;                                          // where the last phase name ended: two names never overlap
    (selected ? selected.phases : []).forEach((p) => {
      if (p.e <= view.ws || p.s >= view.we) return;
      const x0 = Math.max(X0, xOf(p.s)), x1 = Math.min(X0 + W, xOf(p.e));
      if (p.s > view.ws) el("line", { x1: x0, y1: 24, x2: x0, y2: bottom - 4, stroke: "var(--line-strong)", "stroke-width": 0.8, "stroke-dasharray": "4 3" }, svg);
      const width = p.name.length * 6.6;
      const short = x1 - x0 < width + 8;                       // a short phase gets its name ending where it ends, over the empty space on the left
      const left = short ? x1 - 4 - width : x0 + 4;
      if (left < labelEnd + 6) return;                         // no room: the name is left out
      el("text", { class: "muted", x: short ? x1 - 4 : x0 + 4, y: 34, "text-anchor": short ? "end" : "start" }, svg, p.name);
      labelEnd = left + width;
    });
    headers.forEach((h) => el("text", { class: "muted small", x: 16, y: h.y + 12 }, svg, h.label));
    used.forEach((t) => {
      if (!rowY.has(t.id)) return;
      const y0 = rowY.get(t.id);
      el("rect", { class: "box", x: X0, y: y0, width: W, height: 24, rx: 4 }, svg);
      const label = el("text", { class: "muted", x: 16, y: y0 + 12, "dominant-baseline": "central" }, svg, t.name.length > 28 ? t.name.slice(0, 27) + "…" : t.name);
      el("title", {}, label, t.name);
    });
    idle.forEach((g) => {
      if (g.e <= view.ws || g.s >= view.we) return;
      const x0 = Math.max(X0, xOf(g.s)), x1 = Math.min(X0 + W, xOf(g.e));
      const r = el("rect", { class: "idle slice", x: x0, y: rowY.get(g.tid) + 2, width: Math.max(1, x1 - x0), height: 20, rx: 3, fill: "url(#hatch)", stroke: "var(--red-s)", "stroke-width": 0.5 }, svg);
      r.__tip = { idle: g };
    });
    const lots = data.events.length > 4000;                    // with very many events, the ones too thin to see are left out
    data.events.forEach((e) => {
      if (!rowY.has(e.tid) || e.e <= view.ws || e.s >= view.we) return;
      const m = models.get(e.model);
      const x0 = Math.max(X0, xOf(e.s)), x1 = Math.min(X0 + W, xOf(e.e));
      if (lots && x1 - x0 < 0.4 && e.k !== "wait") return;
      const w = Math.max(1, x1 - x0);
      const g = el("g", { class: "slice " + (e.k === "wait" ? "c-red" : m ? m.color : "c-gray") }, svg);
      el("rect", { x: x0, y: rowY.get(e.tid) + 2, width: w, height: 20, rx: 3, "stroke-width": 0.5 }, g);
      if (e.k !== "wait" && e.b !== undefined && m && w >= 26) {
        el("text", { x: x0 + w / 2, y: rowY.get(e.tid) + 12, "text-anchor": "middle", "dominant-baseline": "central" }, g, m.tag + e.b);
      }
      g.__tip = { event: e };
      view.slices.push({ g, s: e.s, model: e.model });
    });
    view.head = el("line", { x1: X0, y1: 24, x2: X0, y2: bottom - 4, stroke: "var(--amber)", "stroke-width": 2, "pointer-events": "none" }, svg);
    let lx = X0;                                               // the legend: the models, waiting, and idle
    const legend = data.models.map((m) => [m.color, m.tag + "  " + m.name]).concat([["c-red", "waiting"]]);
    legend.forEach(([cls, label]) => {
      el("rect", { class: cls, x: lx, y: bottom + 6, width: 12, height: 12, rx: 2, "stroke-width": 0.5 }, svg);
      el("text", { x: lx + 18, y: bottom + 12, "dominant-baseline": "central" }, svg, label);
      lx += 18 + label.length * 6.6 + 16;
    });
    if (idle.length) {
      el("rect", { x: lx, y: bottom + 6, width: 12, height: 12, rx: 2, fill: "url(#hatch)", stroke: "var(--red-s)", "stroke-width": 0.5 }, svg);
      el("text", { x: lx + 18, y: bottom + 12, "dominant-baseline": "central" }, svg, "GPU idle");
    }
  }

  // -- what the pointer is over
  const tip = document.getElementById("tip");
  function tipRows(item) {
    if (item.idle) {
      const g = item.idle, m = models.get(g.model);
      return [m.name + ": the GPU was idle", ["Duration", fmtMs(g.e - g.s)], ["Start", fmtMs(g.s)],
              ["Between", m.tag + g.before + " and " + m.tag + g.after], ["How", "no block ran on this stream (worked out from the runs)"]];
    }
    const e = item.event, m = models.get(e.model), t = trackOf.get(e.tid), b = m && e.b !== undefined ? m.byId.get(e.b) : null;
    const rows = [(m ? m.name + ": " : "") + e.name];
    const add = (a, v) => { if (v !== undefined && v !== null && v !== "") rows.push([a, v]); };
    add("Kind", e.k + (e.reason ? ", for " + e.reason : ""));
    add("Start", fmtMs(e.s));
    add("Duration", fmtMs(e.e - e.s));
    add("Track", t ? t.name + " (" + t.kind.toUpperCase() + " time)" : "");
    add("Block", b ? b.name + ", " + b.tier + " tier" : "");
    add("Pass", e.pass);
    add("Route", e.from ? e.from + (e.via ? " → " + e.via : "") + (e.to ? " → " + e.to : "") : "");
    add("Bytes", e.bytes ? fmtBytes(e.bytes) : "");
    add("Read by", e.source);
    return rows;
  }
  function showTip(ev, item) {
    if (!tip) return;
    tip.replaceChildren();
    const rows = tipRows(item);
    const title = document.createElement("div");
    title.className = "tip-title";
    title.textContent = rows[0];
    tip.appendChild(title);
    rows.slice(1).forEach(([a, v]) => {
      const row = document.createElement("div");
      const k = document.createElement("span");
      k.textContent = a;
      row.append(k, " " + v);
      tip.appendChild(row);
    });
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - w - 8) + "px";
    tip.style.top = Math.min(ev.clientY + 14, innerHeight - h - 8) + "px";
  }
  const hideTip = () => { if (tip) tip.hidden = true; };

  // -- the window: zoom, pan, and where the playhead is
  let pending = null;
  function setWindow(s, e) {
    const span = Math.min(Math.max(e - s, minSpan), dur);
    s = Math.min(Math.max(s, 0), dur - span);
    view.ws = s;
    view.we = s + span;
    view.k = W / span;
    if (hooks.changed) hooks.changed(view.ws, view.we);
    if (!pending) pending = requestAnimationFrame(() => { pending = null; draw(); if (hooks.drawn) hooks.drawn(); });
  }
  const at = (x) => view.ws + (x - X0) / view.k;
  function zoomAt(t, factor) {
    const span = Math.min(Math.max((view.we - view.ws) * factor, minSpan), dur);
    const ratio = (t - view.ws) / (view.we - view.ws);
    setWindow(t - ratio * span, t - ratio * span + span);
  }

  svg.addEventListener("wheel", (ev) => {
    const x = px(ev);
    if (x < X0 || x > X0 + W) return;
    ev.preventDefault();
    zoomAt(at(x), ev.deltaY < 0 ? 0.8 : 1.25);
  }, { passive: false });
  let drag = null;
  svg.addEventListener("pointerdown", (ev) => {
    const x = px(ev);
    if (x < X0 || x > X0 + W) return;
    drag = { x, ws: view.ws, we: view.we, moved: false };
    try { svg.setPointerCapture(ev.pointerId); } catch (err) { /* a pointer that is gone: the drag still works inside the drawing */ }
  });
  svg.addEventListener("pointermove", (ev) => {
    if (drag) {
      const x = px(ev);
      if (Math.abs(x - drag.x) > 3) drag.moved = true;
      if (drag.moved) {
        hideTip();
        const dt = (drag.x - x) / (W / (drag.we - drag.ws));
        setWindow(drag.ws + dt, drag.we + dt);
      }
      return;
    }
    let node = ev.target;
    while (node && node !== svg && !node.__tip) node = node.parentNode;
    if (node && node.__tip) showTip(ev, node.__tip); else hideTip();
  });
  svg.addEventListener("pointerup", (ev) => {
    if (drag && !drag.moved && hooks.seek) hooks.seek(Math.max(0, Math.min(dur, at(px(ev)))));
    drag = null;
  });
  svg.addEventListener("pointerleave", hideTip);

  draw();
  return {
    redraw: draw,
    setWindow,
    fit: () => setWindow(0, dur),
    zoomBy: (factor) => zoomAt((view.ws + view.we) / 2, factor),
    window: () => [view.ws, view.we],
    follow(t) {                                                // a zoomed timeline keeps the playhead in view while it plays
      if (view.we - view.ws >= dur) return;
      if (t > view.we || t < view.ws) setWindow(t - (view.we - view.ws) * 0.15, t + (view.we - view.ws) * 0.85);
    },
    update(t) {
      const x = Math.max(X0, Math.min(X0 + W, xOf(t)));
      view.head.setAttribute("x1", x);
      view.head.setAttribute("x2", x);
      view.head.setAttribute("opacity", t >= view.ws && t <= view.we ? 1 : 0);
      view.slices.forEach((s) => {
        const faded = state.model && s.model !== state.model;   // the other models stay, but quieter
        s.g.style.opacity = t < s.s ? 0.3 : faded ? 0.5 : 1;
      });
    },
  };
}
