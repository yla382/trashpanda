// The page: it fetches the trace, builds the drawings and the panels, and moves them all with one clock.

import { clear } from "./svg.js";
import { normalize } from "./data.js";
import { buildBars } from "./bars.js";
import { buildFiles, traceUrl } from "./files.js";
import { buildMemory } from "./memory.js";
import { buildMemoryChart } from "./memorychart.js";
import { NOT_SHOWN, buildList, buildMemoryNow, buildNow, buildRunTable, buildStrip, buildTotals, showCard } from "./panels.js";
import { buildTimeline } from "./timeline.js";
import { buildUsage } from "./usage.js";

const $ = (id) => document.getElementById(id);
const PER_SECOND = 100;                                          // at 1x, 100 ms of the run pass in a second
const RATES = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128];

buildFiles($("files"));

fetch(traceUrl())
  .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
  .then(({ ok, d }) => {
    if (!ok) throw new Error(d.error);
    start(normalize(d));
  })
  .catch((e) => {
    $("error").textContent = e.message;
    $("error").hidden = false;
  });

function start(data) {
  const state = { model: data.models[0].id };
  const byId = new Map(data.models.map((m) => [m.id, m]));
  const current = () => byId.get(state.model);
  const dur = data.duration;

  $("meta").textContent = data.models.length + (data.models.length === 1 ? " model, " : " models, ") + Math.round(dur) + " ms" +
    (data.note ? " (" + data.note + ")" : "");
  $("view").hidden = false;
  NOT_SHOWN.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    $("not").appendChild(li);
  });

  const select = (id) => { state.model = id; show(); };
  const strip = buildStrip($("strip"), data, state, select);
  const list = buildList($("models"), data, state, select);
  const now = buildNow($("now"), data);
  const usage = buildUsage(data);
  const chart = buildMemoryChart($("mc"), data, usage, { drawn: () => update() });
  const timeline = buildTimeline($("tl"), data, state, { seek, changed: ranged, drawn: () => update() });
  const memNow = buildMemoryNow($("memnow"), data, usage, state);
  buildRunTable($("runtable"), data);
  if (!chart) $("mcwrap").hidden = true;
  if (!usage.has("vram") && !usage.has("ram") && !usage.has("pinned")) $("memnowwrap").hidden = true;
  const memory = buildMemory($("mem"), data, state);
  let bars, totals;

  const scrub = $("scrub");
  scrub.max = dur;
  let t = 0, playing = false, last = 0;
  const speed = $("speed");
  RATES.forEach((r) => {
    const o = document.createElement("option");
    o.value = String(r);
    o.textContent = r + "x";
    speed.appendChild(o);
  });
  const want = dur / 40 / PER_SECOND;                           // a run of about 40 seconds on the screen, if it can be
  let rate = [...RATES].reverse().find((r) => r <= want) || RATES[0];
  if (dur < 20000) rate = Math.min(rate, 0.5);
  speed.value = String(rate);
  speed.onchange = () => { rate = +speed.value; };

  const pass = $("pass");
  pass.onchange = () => { playing = false; setButton(); view(); update(); };

  function show() {                                              // the selected model changed
    const m = current();
    bars = buildBars($("bars"), data, m);
    totals = buildTotals($("totals"), data, m);
    showCard($("card"), data, m);
    clear(pass);
    pass.hidden = m.passes.length < 2;
    if (!pass.hidden) {
      const add = (value, label) => {
        const o = document.createElement("option");
        o.value = value;
        o.textContent = label;
        pass.appendChild(o);
      };
      add("all", "Whole run");
      m.passes.forEach((p, i) => add(String(i), p.name));
    }
    timeline.redraw();
    view();
    update();
  }

  function view() {
    const m = current();
    const one = pass.hidden || pass.value === "all" ? null : m.passes[+pass.value];
    const w = one ? { s: one.s, e: one.e } : { s: 0, e: dur };
    timeline.setWindow(w.s, w.e);
    bars.setWindow(w.s, w.e);
    if (one && (t < w.s || t >= w.e)) t = w.s;
  }

  function update() {
    const m = current();
    scrub.value = t;
    const phase = m.phases.find((p) => t >= p.s && t < p.e);
    $("clock").textContent = "t = " + Math.round(t) + " ms" + (phase ? ", " + m.tag + " " + phase.name : "");
    strip.update(t);
    list.update(t);
    memory.update(t);
    timeline.update(t);
    bars.update(t);
    now.update(t);
    memNow.update(t);
    if (chart) chart.update(t);
    totals.update(t);
  }

  function seek(at) {                                           // the timeline was clicked
    playing = false;
    setButton();
    t = at;
    update();
  }

  function ranged(ws, we) {
    if (chart) chart.setWindow(ws, we);
    $("range").textContent = Math.round(ws) + " to " + Math.round(we) + " ms (" + Math.round(we - ws) + " ms shown)";
  }

  function setButton() {
    $("play").textContent = playing ? "Pause" : "Play";
  }

  function frame(at) {
    if (!playing) return;
    t = Math.min(dur, t + ((at - last) / 1000) * PER_SECOND * rate);
    last = at;
    update();
    timeline.follow(t);
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
  $("zin").onclick = () => timeline.zoomBy(0.6);
  $("zout").onclick = () => timeline.zoomBy(1.6);
  $("fit").onclick = () => {
    if (!pass.hidden) pass.value = "all";
    view();
    timeline.fit();
  };
  setButton();
  show();
}
