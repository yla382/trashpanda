// Memory in use over time: VRAM, RAM and pinned memory, stacked by model, on the same time axis as the timeline (it follows
// the timeline's window). A dashed line in a model's color is the budget its plan counted on, over the time the model lives.

import { clear, el } from "./svg.js";
import { fmtBytes, fmtMs } from "./data.js";

const X0 = 196, W = 444, H = 54, GAP = 26, TOP = 8;
const NAMES = { vram: "VRAM", ram: "RAM", pinned: "Pinned" };

export function buildMemoryChart(svg, data, usage, hooks = {}) {
  const resources = Object.keys(NAMES).filter((r) => usage.has(r));
  if (!resources.length) return null;
  const height = TOP + resources.length * (H + GAP) + 4;
  svg.setAttribute("viewBox", "0 0 680 " + height);
  const view = { ws: 0, we: data.duration, head: null };
  const xOf = (t) => X0 + ((t - view.ws) / (view.we - view.ws)) * W;
  const budgetOf = (m, r) => (r === "vram" ? m.plan.vram_budget : r === "ram" ? m.plan.ram_budget : null);
  const peaks = Object.fromEntries(resources.map((r) => [r, Math.max(usage.peak(r), ...data.models.map((m) => budgetOf(m, r) || 0), 1)]));
  const top = (i) => TOP + i * (H + GAP);

  function draw() {
    clear(svg);
    resources.forEach((r, i) => {
      const y0 = top(i), max = peaks[r] * 1.08;
      const yOf = (v) => y0 + H - (v / max) * H;
      el("text", { class: "th", x: 16, y: y0 + 14 }, svg, NAMES[r]);
      el("text", { class: "muted small", x: 16, y: y0 + 30 }, svg, "up to " + fmtBytes(peaks[r]));
      el("text", { class: "muted small", x: 16, y: y0 + 44 }, svg, usage.measured(r) ? "measured" : "worked out from the events");
      el("rect", { class: "box", x: X0, y: y0, width: W, height: H, rx: 4 }, svg);
      const s = usage.series(r, view.ws, view.we);
      let below = s.times.map(() => 0);
      usage.owners.forEach((m) => {
        const values = s.per.get(m.key);
        if (!values.some((v) => v > 0)) return;
        const above = below.map((b, k) => b + values[k]);
        let d = "M" + xOf(s.times[0]) + " " + yOf(above[0]);
        for (let k = 1; k < s.times.length; k++) d += " L" + xOf(s.times[k]) + " " + yOf(above[k]);
        for (let k = s.times.length - 1; k >= 0; k--) d += " L" + xOf(s.times[k]) + " " + yOf(below[k]);
        const path = el("path", { d: d + " Z", "stroke-width": 0.5 }, svg);
        path.setAttribute("style", "fill:var(" + m.fill + ");stroke:var(" + m.stroke + ")");
        below = above;
      });
      data.models.forEach((m) => {                             // the budget of each model, while the model lives
        const b = budgetOf(m, r);
        if (!b || m.e <= view.ws || m.s >= view.we) return;
        const line = el("line", { x1: Math.max(X0, xOf(m.s)), x2: Math.min(X0 + W, xOf(m.e)), y1: yOf(b), y2: yOf(b), "stroke-dasharray": "5 3", "stroke-width": 1.2 }, svg);
        line.setAttribute("style", "stroke:var(--" + m.color + "-s)");
        el("title", {}, line, m.name + ": the " + NAMES[r] + " budget of its plan, " + fmtBytes(b));
      });
    });
    view.head = el("line", { x1: X0, y1: TOP, x2: X0, y2: height - 4, stroke: "var(--amber)", "stroke-width": 2, "pointer-events": "none" }, svg);
  }

  const tip = document.getElementById("tip");
  svg.addEventListener("pointermove", (ev) => {
    if (!tip) return;
    const r = svg.getBoundingClientRect(), x = ((ev.clientX - r.left) * 680) / r.width, y = ((ev.clientY - r.top) * 680) / r.width;
    if (x < X0 || x > X0 + W) { tip.hidden = true; return; }
    const i = Math.min(resources.length - 1, Math.max(0, Math.floor((y - TOP) / (H + GAP))));
    const t = view.ws + ((x - X0) / W) * (view.we - view.ws), r0 = resources[i], now = usage.at(r0, t);
    tip.replaceChildren();
    const title = document.createElement("div");
    title.className = "tip-title";
    title.textContent = NAMES[r0] + " at " + fmtMs(t) + ": " + fmtBytes(now.total);
    tip.appendChild(title);
    usage.owners.forEach((m) => {
      const row = now.byModel.get(m.key);
      if (!row) return;
      const line = document.createElement("div");
      const k = document.createElement("span");
      k.textContent = m.tag + " " + m.name;
      line.append(k, " " + fmtBytes(row.total) + "  (" + Object.entries(row.kinds).map(([a, b]) => a + " " + fmtBytes(b)).join(", ") + ")");
      tip.appendChild(line);
    });
    tip.hidden = false;
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = Math.min(ev.clientY + 14, innerHeight - tip.offsetHeight - 8) + "px";
  });
  svg.addEventListener("pointerleave", () => { if (tip) tip.hidden = true; });

  draw();
  return {
    setWindow(ws, we) {
      view.ws = ws;
      view.we = we;
      draw();
      if (hooks.drawn) hooks.drawn();
    },
    update(t) {
      const x = Math.max(X0, Math.min(X0 + W, xOf(t)));
      view.head.setAttribute("x1", x);
      view.head.setAttribute("x2", x);
      view.head.setAttribute("opacity", t >= view.ws && t <= view.we ? 1 : 0);
    },
  };
}
