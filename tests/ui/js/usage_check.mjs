// Prints the memory in use that the page works out (static/usage.js) for a trace, at the times given.
//   node usage_check.mjs <trace.json> <time in ms> ...
import { readFileSync } from "node:fs";

import { normalize } from "../../../src/trash_panda/ui/static/data.js";
import { buildUsage } from "../../../src/trash_panda/ui/static/usage.js";

const [path, ...times] = process.argv.slice(2);
const data = normalize(JSON.parse(readFileSync(path, "utf8")));
const usage = buildUsage(data);
const resources = ["vram", "ram", "pinned"];
const rows = times.map((t) => {
  const row = { t: +t };
  for (const r of resources) {
    const now = usage.at(r, +t);
    row[r] = { total: now.total, models: Object.fromEntries([...now.byModel].map(([m, v]) => [m, v])) };
  }
  return row;
});
console.log(JSON.stringify({
  rows,
  peaks: Object.fromEntries(resources.map((r) => [r, usage.peak(r)])),
  has: Object.fromEntries(resources.map((r) => [r, usage.has(r)])),
  measured: Object.fromEntries(resources.map((r) => [r, usage.measured(r)])),
  series: usage.series("vram", 0, data.duration, 4).totals,
}));
