// Shared helpers for the paper's web figures (D3). Every figure page loads d3, this file and its own data
// file (data/<name>.js, which sets window.FIG), draws into <svg id="fig">, and is printed to PDF by
// scripts/make_web_figures.py. Sizes are CSS px (1 px = 0.75 pt); the page size is the figure size.

const css = getComputedStyle(document.documentElement);
const col = (name) => css.getPropertyValue(`--${name}`).trim();
// Sunset (CARTO), for ordered quantities (denoising progress, densities): ramp(0) light ... ramp(1) dark.
const ramp = (u) => d3.interpolateRgbBasis([0, 1, 2, 3, 4, 5].map((i) => col(`ramp${i}`)))(Math.max(0, Math.min(1, u)));

function setup(width, height) {
  const st = document.createElement("style");
  st.textContent = `@page { size: ${width}px ${height}px; margin: 0; } body { width: ${width}px; height: ${height}px; }`;
  document.head.appendChild(st);
  return d3.select("#fig").attr("width", width).attr("height", height).attr("viewBox", `0 0 ${width} ${height}`);
}

// A panel: a group translated to (x, y) with an inner plotting area after margins.
function panel(svg, x, y, w, h, m = { t: 30, r: 8, b: 26, l: 34 }) {
  const g = svg.append("g").attr("transform", `translate(${x},${y})`);
  const inner = g.append("g").attr("transform", `translate(${m.l},${m.t})`);
  return { g, inner, w: w - m.l - m.r, h: h - m.t - m.b, m };
}

// Title that states the finding, with an optional letter and a one-line subtitle.
function title(p, letter, text, sub) {
  const t = p.g.append("text").attr("class", "title").attr("x", 0).attr("y", 10);
  if (letter) t.append("tspan").attr("class", "letter").text(`(${letter}) `);
  t.append("tspan").text(text);
  if (sub) p.g.append("text").attr("class", "subtitle").attr("x", 0).attr("y", 22).text(sub);
}

function axes(p, x, y, o = {}) {
  const gy = p.inner.append("g").attr("class", "grid");
  gy.selectAll("line").data(y.ticks(o.yTicks || 4)).join("line")
    .attr("x1", 0).attr("x2", p.w).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
  const ax = d3.axisBottom(x).tickSizeOuter(0).tickSize(3);
  if (o.xTickValues) ax.tickValues(o.xTickValues);
  if (o.xFormat) ax.tickFormat(o.xFormat);
  p.inner.append("g").attr("class", "axis").attr("transform", `translate(0,${p.h})`).call(ax);
  const ay = d3.axisLeft(y).ticks(o.yTicks || 4).tickSizeOuter(0).tickSize(0).tickPadding(4);
  if (o.yFormat) ay.tickFormat(o.yFormat);
  p.inner.append("g").attr("class", "axis").call(ay).call((s) => s.select(".domain").remove());
  if (o.xLabel) p.inner.append("text").attr("class", "axis-label").attr("x", p.w / 2).attr("y", p.h + 22)
    .attr("text-anchor", "middle").text(o.xLabel);
  if (o.yLabel) p.inner.append("text").attr("class", "axis-label").attr("transform", `translate(${-26},${p.h / 2}) rotate(-90)`)
    .attr("text-anchor", "middle").text(o.yLabel);
}

// A line with dots, and its label written at the end of the line instead of in a legend.
function series(p, x, y, pts, color, label, o = {}) {
  const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
  p.inner.append("path").attr("d", line(pts)).attr("fill", "none").attr("stroke", color)
    .attr("stroke-width", o.width || 1.6).attr("stroke-dasharray", o.dash || null);
  p.inner.selectAll(null).data(pts).join("circle").attr("cx", (d) => x(d[0])).attr("cy", (d) => y(d[1]))
    .attr("r", o.r || 2.6).attr("fill", color).attr("fill-opacity", 0.35).attr("stroke", color).attr("stroke-width", 1);
  if (label) (p.labels = p.labels || []).push({ x: x(pts[pts.length - 1][0]) + 6, y: y(pts[pts.length - 1][1]), text: label, color });
}

// Write the end-of-line labels collected by series(), pushed apart so they never overlap.
function placeLabels(p, gap = 10) {
  const L = (p.labels || []).sort((a, b) => a.y - b.y);
  for (let it = 0; it < 50; it++) {
    let moved = false;
    for (let i = 1; i < L.length; i++) {
      const d = L[i].y - L[i - 1].y;
      if (d < gap) { const s = (gap - d) / 2; L[i].y += s; L[i - 1].y -= s; moved = true; }
    }
    if (!moved) break;
  }
  L.forEach((l) => p.inner.append("text").attr("class", "direct").attr("x", l.x).attr("y", l.y + 3).style("fill", l.color).text(l.text));
}

// A band (e.g. a 95% interval) under a series.
function band(p, x, y, pts, color, opacity = 0.15) {
  const area = d3.area().x((d) => x(d[0])).y0((d) => y(d[1])).y1((d) => y(d[2]));
  p.inner.append("path").attr("d", area(pts)).attr("fill", color).attr("opacity", opacity);
}
