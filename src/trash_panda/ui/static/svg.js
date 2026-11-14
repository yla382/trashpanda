// Small helpers for building SVG.

const NS = "http://www.w3.org/2000/svg";

export function el(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

export function place(e, p) {
  e.style.transform = "translate(" + p[0] + "px," + p[1] + "px)";
}

// A tick step that gives at most `most` ticks over `span`.
export function tickStep(span, most = 8) {
  const steps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];
  return steps.find((s) => span / s <= most) || 100000;
}
