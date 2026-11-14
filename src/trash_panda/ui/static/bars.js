// Where each block of the selected model spent its time: a stacked bar for each block, from the events of the window.
// A block's running is in the color of its model; reading and copying are gray, waiting is red.

import { clear, el } from "./svg.js";

const STACK = ["read", "load", "stage", "copy", "wait", "call", "run"];     // the order of a block's time in its bar
const KIND = { read: "c-gray", load: "c-gray", stage: "c-gray2", copy: "c-gray2", wait: "c-red" };

export function buildBars(svg, data, model) {
  clear(svg);
  const rowY = (i) => 28 + i * 22;
  const lastEnd = data.events.reduce((a, e) => Math.max(a, e.e), 0);
  const view = { ws: 0, we: Infinity };
  svg.setAttribute("viewBox", "0 0 680 " + (rowY(model.blocks.length) + 40));

  function draw() {
    clear(svg);
    const whole = view.ws <= 0 && view.we >= lastEnd - 1e-6;
    const mine = model.events.filter((e) => e.s >= view.ws && e.s < view.we && e.b !== undefined);
    const workRun = model.workRun;
    const sums = model.blocks.map((b) => {
      const own = mine.filter((e) => e.b === b.id);
      const parts = STACK.filter((k) => k !== "run" && k !== "call" || k === workRun)
        .flatMap((k) => own.filter((e) => e.k === k).map((e) => ({ k, ms: e.e - e.s })));
      return { b, parts, total: parts.reduce((a, p) => a + p.ms, 0) };
    });
    const scale = 250 / Math.max(1, ...sums.map((s) => s.total));
    el("text", { class: "muted", x: 16, y: 12 }, svg, "Where each block's time went, in " + (whole ? "the whole run" : "this pass"));
    view.hl = el("rect", { x: 8, y: 0, width: 400, height: 20, rx: 4, opacity: 0 }, svg);
    sums.forEach((s, i) => {
      el("text", { x: 16, y: rowY(i) + 8, "dominant-baseline": "central" }, svg, model.tag + s.b.id);
      let x = 56;
      s.parts.forEach((p) => {
        const cls = p.k === workRun ? model.color : KIND[p.k];
        el("rect", { class: cls, x, y: rowY(i), width: Math.max(p.ms * scale, 0.5), height: 16, "stroke-width": 0.5 }, svg);
        x += p.ms * scale;
      });
      el("text", { class: "muted", x: x + 6, y: rowY(i) + 8, "dominant-baseline": "central" }, svg, Math.round(s.total) + " ms");
    });
    const y = rowY(model.blocks.length) + 8;                   // the legend
    [[model.color, "running"], ["c-gray", "read from disk"], ["c-gray2", "copied"], ["c-red", "waiting"]].forEach(([cls, label], i) => {
      el("rect", { class: cls, x: 56 + i * 120, y, width: 12, height: 12, rx: 2, "stroke-width": 0.5 }, svg);
      el("text", { x: 74 + i * 120, y: y + 6, "dominant-baseline": "central" }, svg, label);
    });
    view.rowY = rowY;
  }
  draw();
  return {
    setWindow(s, e) {
      view.ws = s;
      view.we = e;
      draw();
    },
    update(t) {
      const cur = model.events.find((e) => (e.k === "wait" || e.k === model.workRun) && e.b !== undefined && t >= e.s && t < e.e);
      const row = cur ? model.blocks.findIndex((b) => b.id === cur.b) : -1;
      if (cur && row >= 0) {
        view.hl.setAttribute("y", view.rowY(row) - 2);
        view.hl.setAttribute("opacity", 0.18);
        view.hl.style.fill = cur.k === "wait" ? "var(--red-s)" : "var(--amber)";
      } else {
        view.hl.setAttribute("opacity", 0);
      }
    },
  };
}
