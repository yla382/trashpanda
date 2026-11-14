// Everything of the page that is not a drawing of the memory or the timeline: the strip of models, the list of models,
// the card of the selected one, what happens at the playhead, the totals, and the list of what the viewer does not show.

import { clear, el, tickStep } from "./svg.js";
import { fmtBytes, fmtMs, STARTED } from "./data.js";

// -- the strip: one lane for each model, on one time axis

export function buildStrip(svg, data, state, select) {
  const LABEL = 210, X0 = LABEL + 10, W = 1000 - X0 - 10, LANE = 44;
  const k = W / data.duration;
  const height = 24 + data.models.length * LANE + 8;
  svg.setAttribute("viewBox", "0 0 1000 " + height);
  const step = tickStep(data.duration);
  for (let m = 0; m <= data.duration + 1e-6; m += step) {
    el("text", { class: "muted", x: X0 + m * k, y: 12, "text-anchor": "middle" }, svg, m === 0 ? "0 ms" : String(m));
  }
  const lanes = new Map();
  data.models.forEach((m, i) => {
    const y = 20 + i * LANE;
    const lane = el("g", { class: "lane", tabindex: 0, role: "button", "aria-label": "Show " + m.name }, svg);
    const outline = el("rect", { x: 4, y: y - 2, width: 992, height: LANE - 2, rx: 6, fill: "none" }, lane);
    el("rect", { class: "box", x: X0, y, width: W, height: LANE - 6, rx: 4, "stroke-width": 0.5 }, lane);
    const label = el("text", { class: "th", x: 14, y: y + 12, "dominant-baseline": "central" }, lane, m.tag + "  " + m.name);
    el("text", { class: "muted", x: 14, y: y + 30, "dominant-baseline": "central" }, lane, STARTED[m.startedBy] || m.startedBy);
    const parts = m.phases.length ? m.phases : [{ name: "run", s: m.s, e: m.e }];
    parts.forEach((p) => {
      const x = X0 + p.s * k, w = Math.max(1.5, (p.e - p.s) * k);
      const g = el("g", { class: m.color }, lane);
      el("title", {}, g, m.name + ": " + p.name + ", " + fmtMs(p.e - p.s));
      const r = el("rect", { x, y: y + 4, width: w, height: LANE - 16, rx: 3, "stroke-width": 0.5 }, g);
      if (p.name === "load") r.setAttribute("opacity", 0.55);
      if (w >= p.name.length * 6.6 + 8) el("text", { x: x + w / 2, y: y + 14, "text-anchor": "middle", "dominant-baseline": "central" }, g, p.name);
    });
    m.events.filter((e) => e.k === "preload").forEach((e) => {          // loaded in the background, before the model was entered
      const x = X0 + e.s * k, w = Math.max(1.5, (e.e - e.s) * k);
      const g = el("g", { class: m.color }, lane);
      el("title", {}, g, m.name + ": loaded in the background while the model before it ran, " + fmtMs(e.e - e.s));
      const r = el("rect", { x, y: y + 4, width: w, height: LANE - 16, rx: 3, "stroke-width": 0.8, "stroke-dasharray": "3 2", opacity: 0.45 }, g);
      void r;
    });
    lane.addEventListener("click", () => select(m.id));
    lane.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(m.id); } });
    lanes.set(m.id, outline);
    void label;
  });
  const head = el("line", { x1: X0, y1: 16, x2: X0, y2: height - 4, stroke: "var(--amber)", "stroke-width": 2, "pointer-events": "none" }, svg);
  return {
    update(t) {
      head.setAttribute("x1", X0 + t * k);
      head.setAttribute("x2", X0 + t * k);
      lanes.forEach((outline, id) => {
        const on = id === state.model;
        outline.setAttribute("stroke", on ? "var(--amber)" : "none");
        outline.setAttribute("stroke-width", on ? 1.5 : 0);
      });
    },
  };
}

// -- the list of models and the card of the selected one

function stateOf(model, t) {
  const phase = model.phases.find((p) => t >= p.s && t < p.e);
  if (phase) return phase.name === "load" ? "loading" : "running, " + phase.name;
  return t < model.s ? "not started" : t >= model.e ? "done" : "between passes";
}

export function buildList(host, data, state, select) {
  clear(host);
  const rows = data.models.map((m) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "mrow";
    b.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true"><g class="' + m.color + '"><rect x="1" y="1" width="14" height="14" rx="3" stroke-width="0.5"></rect></g></svg>' +
      '<span class="mname"></span><span class="mstate"></span>';
    b.querySelector(".mname").textContent = m.tag + "  " + m.name;
    b.onclick = () => select(m.id);
    host.appendChild(b);
    return [m, b];
  });
  return {
    update(t) {
      rows.forEach(([m, b]) => {
        b.setAttribute("aria-pressed", String(m.id === state.model));
        b.querySelector(".mstate").textContent = stateOf(m, t);
      });
    },
  };
}

export function showCard(host, data, model) {
  clear(host);
  const rows = [];
  const add = (label, value) => { if (value !== null && value !== undefined && value !== "") rows.push([label, value]); };
  const p = model.plan, o = model.options;
  add("Started by", STARTED[model.startedBy] || model.startedBy);
  if (model.checkpoint) add("Checkpoint", (model.checkpoint.path || "").split(/[\\/]/).pop() + (model.checkpoint.bytes ? ", " + fmtBytes(model.checkpoint.bytes) : ""));
  if (model.dtypes) add("Stored as", model.dtypes.stored + (model.dtypes.compute ? ", expanded to " + model.dtypes.compute + " when a block runs" : ""));
  add("Blocks", model.blocks.length + ": " + model.tiers.gpu + " on the GPU, " + model.tiers.ram + " in RAM, " + model.tiers.disk + " on disk");
  add("Outside the blocks", model.outside ? fmtBytes(model.outside) + ", on the GPU" : "");
  add("Passes", model.passes.length || "");
  add("VRAM budget", p.vram_budget ? fmtBytes(p.vram_budget) + (p.headroom_bytes ? ", less " + fmtBytes(p.headroom_bytes) + " for activations" : "") : "");
  add("RAM budget", p.ram_budget ? fmtBytes(p.ram_budget) : "");
  add("Look-ahead", p.lookahead ? p.lookahead + " blocks" : "");
  const pool = data.pools.find((x) => x.id === model.pool);
  add("Pinned buffers", pool ? pool.buffers + (pool.kind === "shared" ? ", a pool shared with " + pool.models.filter((x) => x !== model.id).join(", ") : ", its own pool") : "none");
  const flags = [];
  if (o.overlap) flags.push("overlap");
  if (o.ram_in_memory === false) flags.push("RAM tier mapped"); else if (o.ram_in_memory) flags.push("RAM tier read into memory");
  if (o.adaptive_lookahead) flags.push("adaptive look-ahead");
  add("Options", flags.join(", "));
  const dl = document.createElement("dl");
  dl.className = "kv";
  rows.forEach(([a, b]) => {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = a;
    dd.textContent = String(b);
    dl.append(dt, dd);
  });
  host.appendChild(dl);
  (p.warnings || []).forEach((w) => {
    const warn = document.createElement("p");
    warn.className = "warn";
    warn.textContent = w;
    host.appendChild(warn);
  });
}

// -- at the playhead: what is happening, and for which model

function say(e, model) {
  const n = "block " + e.b;
  switch (e.k) {
    case "run": return n + " runs on the GPU";
    case "call": return n + " is launched";
    case "wait":
      if (e.reason === "loader") return "waits for " + n + " to be loaded";
      if (e.reason === "pinned buffer") return "the worker waits for a free pinned buffer (for " + n + ")";
      if (e.reason === "gpu") return "waits for the GPU to catch up";
      if (e.reason === "preload") return "waits for the next model to be ready";
      return "waits for " + n + " to be copied";
    case "load": return n + " is read from disk into ordinary memory and copied to the GPU";
    case "read": return e.to === "ram" ? n + " is read from disk into RAM" : n + " is read from disk into a pinned buffer";
    case "stage": return n + " is copied from RAM into a pinned buffer";
    case "copy": return n + " is copied from a pinned buffer to the GPU";
    case "move": return n + " moves from " + e.from + " to " + e.to;
    case "drop": return n + " is dropped";
    case "preload": return "the next model is loaded in the background";
    default: return e.name || e.k;
  }
}

export function buildNow(host, data) {
  const order = { run: 0, call: 1, wait: 2, copy: 3, stage: 4, read: 5, load: 6 };
  const byId = new Map(data.models.map((m) => [m.id, m]));
  return {
    update(t) {
      clear(host);
      const active = data.events.filter((e) => t >= e.s && t < e.e).sort((a, b) => (order[a.k] ?? 9) - (order[b.k] ?? 9));
      if (!active.length) {
        host.textContent = t >= data.duration ? "The run is done." : "Nothing is moving.";
        return;
      }
      const ul = document.createElement("ul");
      ul.className = "now";
      active.slice(0, 10).forEach((e) => {
        const m = byId.get(e.model);
        const li = document.createElement("li");
        if (e.k === "wait") li.className = "stall";
        const tag = document.createElement("span");
        tag.className = "tagpill " + (m ? m.color : "");
        tag.textContent = m ? m.tag : "?";
        li.append(tag, " " + (m && e.pass ? "pass " + e.pass + ": " : "") + say(e, m));
        ul.appendChild(li);
      });
      host.appendChild(ul);
      if (active.length > 10) host.appendChild(document.createTextNode("and " + (active.length - 10) + " more"));
    },
  };
}

// -- totals of the selected model: the main thread's time, split

export function buildTotals(host, data, model) {
  const overlap = (list, t) => list.reduce((a, e) => a + Math.max(0, Math.min(t, e.e) - e.s), 0);
  const loads = model.phases.filter((p) => p.name === "load");
  const cpu = new Set(data.tracks.filter((x) => x.kind === "cpu").map((x) => x.id));
  const runs = model.events.filter((e) => e.k === model.workRun);          // what the main thread did with each block
  const waits = model.events.filter((e) => e.k === "wait" && e.reason !== "pinned buffer" && cpu.has(e.tid));
  return {
    update(t) {
      clear(host);
      const upTo = Math.min(t, model.e);
      const elapsed = Math.max(0, upTo - model.s);
      const load = overlap(loads, upTo), run = overlap(runs, upTo), wait = overlap(waits, upTo);
      const other = elapsed - load - run - wait;
      const dl = document.createElement("dl");
      dl.className = "kv";
      [["Since it started", fmtMs(elapsed)], ["Loading before the run", fmtMs(load)], ["Blocks running", fmtMs(run)],
       ["Waiting", fmtMs(wait)], ["Everything else", fmtMs(Math.max(other, 0))]].forEach(([a, b]) => {
        const dt = document.createElement("dt"), dd = document.createElement("dd");
        dt.textContent = a;
        dd.textContent = b;
        dl.append(dt, dd);
      });
      host.appendChild(dl);
      const note = document.createElement("p");
      note.className = "muted-note";
      note.textContent = other < -1 ? "Events overlap on the main thread, so these do not add up." : "Everything else is time outside the blocks, such as between the passes.";
      host.appendChild(note);
    },
  };
}

// -- what the viewer does not show yet (docs/design/trace-viewer.md)

export const NOT_SHOWN = [
  "A mapped RAM block's first read from the file",
  "Kept blocks that are expanded on every run (counted in the memory in use, not drawn as tiles)",
  "release_gpu and restore_gpu",
  "The weights outside the blocks (counted in the memory in use, not drawn as tiles)",
  "Activations, and what other programs use",
];

// -- memory in use now, from usage.js: each resource, and each model's part of it, against the budget of its plan

export function buildMemoryNow(host, data, usage, state) {
  const names = { vram: "VRAM", ram: "RAM", pinned: "Pinned memory" };
  const resources = Object.keys(names).filter((r) => usage.has(r));
  const peaks = Object.fromEntries(resources.map((r) => [r, Math.max(usage.peak(r), 1)]));
  const budgetOf = (m, r) => (r === "vram" ? m.plan.vram_budget : r === "ram" ? m.plan.ram_budget : null);
  return {
    update(t) {
      clear(host);
      resources.forEach((r) => {
        const now = usage.at(r, t);
        const head = document.createElement("div");
        head.className = "mhead";
        const label = document.createElement("span"), total = document.createElement("b");
        label.textContent = names[r];
        total.textContent = fmtBytes(now.total);
        head.append(label, total);
        host.appendChild(head);
        const bar = document.createElement("div");
        bar.className = "stack";
        usage.owners.forEach((m) => {
          const row = now.byModel.get(m.key);
          if (!row || !row.total) return;
          const seg = document.createElement("span");
          seg.className = "seg " + m.color;
          seg.style.width = (100 * row.total) / peaks[r] + "%";
          seg.title = m.name + ": " + fmtBytes(row.total);
          bar.appendChild(seg);
        });
        host.appendChild(bar);
        usage.owners.forEach((m) => {
          const row = now.byModel.get(m.key);
          if (!row || !row.total) return;
          const line = document.createElement("div");
          line.className = "mline" + (m.key === state.model ? " sel" : "");
          const tag = document.createElement("span");
          tag.className = "tagpill " + m.color;
          tag.textContent = m.tag;
          const model = data.models.find((x) => x.id === m.key);
          const budget = model ? budgetOf(model, r) : null;
          line.append(tag, " " + fmtBytes(row.total) + (budget ? " of a " + fmtBytes(budget) + " budget" : model ? "" : " (" + m.name + ")"));
          host.appendChild(line);
          if (m.key === state.model && Object.keys(row.kinds).length > 1) {
            const kinds = document.createElement("div");
            kinds.className = "mkinds";
            kinds.textContent = Object.entries(row.kinds).map(([a, b]) => a.replace("pool:", "pool ") + " " + fmtBytes(b)).join(", ");
            host.appendChild(kinds);
          }
        });
        if (!now.total) host.appendChild(document.createTextNode("nothing yet"));
      });
      const note = document.createElement("p");
      note.className = "muted-note";
      note.textContent = "Worked out from the events. Activations, and the pages of a mapped RAM tier, are not in the trace.";
      host.appendChild(note);
    },
  };
}

// -- the whole run, each model: how its time went

export function buildRunTable(host, data) {
  const cpu = new Set(data.tracks.filter((x) => x.kind === "cpu").map((x) => x.id));
  const sum = (list) => list.reduce((a, e) => a + (e.e - e.s), 0);
  const table = document.createElement("table");
  table.className = "runtable";
  const head = document.createElement("tr");
  ["Model", "Loading", "Running", "Waiting", "Else"].forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    head.appendChild(th);
  });
  const thead = document.createElement("thead");
  thead.appendChild(head);
  table.appendChild(thead);
  const body = document.createElement("tbody");
  data.models.forEach((m) => {
    const load = sum(m.phases.filter((p) => p.name === "load"));
    const run = sum(m.events.filter((e) => e.k === m.workRun));
    const wait = sum(m.events.filter((e) => e.k === "wait" && e.reason !== "pinned buffer" && cpu.has(e.tid)));
    const other = Math.max(0, m.e - m.s - load - run - wait);
    const tr = document.createElement("tr");
    const first = document.createElement("td");
    const tag = document.createElement("span");
    tag.className = "tagpill " + m.color;
    tag.textContent = m.tag;
    first.append(tag, " " + m.name);
    tr.appendChild(first);
    [load, run, wait, other].forEach((v) => {
      const td = document.createElement("td");
      td.textContent = fmtMs(v);
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
  table.appendChild(body);
  host.appendChild(table);
}
