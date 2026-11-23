// The probing diagram (Figure 2), shared by fig_setup.html and its comparison sheets. The page defines `svg`
// (from setup()) and `F` (FIG.film of data/fig_steps_fall.js) before loading this file, then calls figure().
const C = { ink: col("ink"), muted: col("muted"), est: col("est"), lay: col("layer"), lay2: col("layer2"), mark: col("mark"),
            dim: col("dim"), neu: col("neutral"), axis: col("axis"), sec: col("second") };
const tint = (c, t) => d3.interpolateRgb("#ffffff", c)(t), shade = (c, t) => d3.interpolateRgb(c, "#000000")(t);
const noise = (i) => Math.abs((Math.sin(i * 12.9898 + 4.1) * 43758.5453) % 1), q = (v) => Math.round(v * 4) / 4;
const defs = svg.append("defs"); let gid = 0; const gcache = {};
function grad(stops, x2 = 0, y2 = 1) {
  const key = JSON.stringify([stops, x2, y2]); if (gcache[key]) return gcache[key];
  const id = `g${gid++}`, lg = defs.append("linearGradient").attr("id", id).attr("x1", 0).attr("y1", 0).attr("x2", x2).attr("y2", y2);
  stops.forEach(([o, c]) => lg.append("stop").attr("offset", o).attr("stop-color", c));
  return (gcache[key] = `url(#${id})`);
}

// text in the paper's notation: $..$ italic, #..# bold italic (vectors), _{..} subscript, U+0302 a drawn hat
const hats = [];
function tx(g, x, y, s, o = {}) {
  const size = o.size || 9, color = o.color || C.muted;
  const t = g.append("text").attr("x", x).attr("y", y).attr("text-anchor", o.anchor || "start")
    .style("fill", color).style("font-size", `${size}px`).style("font-weight", o.bold ? 650 : 400);
  let it = !!o.italic, vec = false, sub = false, dy = 0, n = 0;
  s.split(/(\$|#|_\{|\})/).filter((p) => p !== "").forEach((p) => {
    if (p === "$") it = !it; else if (p === "#") vec = !vec;
    else if (p === "_{") { sub = true; dy += 2.4; } else if (p === "}") { sub = false; dy -= 2.4; }
    else {
      let txt = "";
      for (const ch of p) { if (ch === "̂") hats.push({ g, t, i: n - 1, size, color }); else { txt += ch; n += 1; } }
      const sp = t.append("tspan").attr("dy", dy).text(txt).style("font-style", it || vec ? "italic" : "normal");
      if (vec) sp.style("font-weight", 650).style("stroke", color).style("stroke-width", "0.35px");
      if (sub) sp.style("font-size", `${size * 0.75}px`);
      dy = 0;
    }
  });
  return t;
}
function drawHats() {
  hats.forEach(({ g, t, i, size, color }) => {
    const p = t.node().getStartPositionOfChar(i), w = t.node().getExtentOfChar(i).width, sh = size * 0.1;
    const yb = p.y - size * 0.5 - 1.1, ya = yb - 2.1;
    g.append("path").attr("d", `M${p.x + 0.22 * w + sh},${yb} L${p.x + 0.55 * w + sh},${ya} L${p.x + 0.88 * w + sh},${yb}`)
      .attr("fill", "none").attr("stroke", color).attr("stroke-width", 0.75).attr("stroke-linecap", "round");
  });
}

// shared pieces: a soft shadow, a card, a curved link with a slim chevron, a shaded token tile, an isometric slab
function shadow(g, x, y, w, h, r) {
  [4.5, 3, 1.8, 0.8].forEach((k) => g.append("rect").attr("x", x - k).attr("y", y - k + 2.2).attr("width", w + 2 * k)
    .attr("height", h + 2 * k).attr("rx", r + k).attr("fill", "#1c2b4a").attr("opacity", 0.028));
}
function card(g, x, y, w, h, o = {}) {
  if (o.shadow !== false) shadow(g, x, y, w, h, o.r ?? 5);
  return g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("rx", o.r ?? 5)
    .attr("fill", o.fill || "#fff").attr("stroke", o.stroke || "none").attr("stroke-width", o.sw || 1);
}
function link(g, a, b, c, o = {}) {
  const [x0, y0] = a, [x1, y1] = b, v = o.mode === "v";
  const c1 = v ? [x0, (y0 + y1) / 2] : [(x0 + x1) / 2, y0], c2 = v ? [x1, (y0 + y1) / 2] : [(x0 + x1) / 2, y1];
  g.append("path").attr("d", `M${x0},${y0} C${c1} ${c2} ${x1},${y1}`).attr("fill", "none").attr("stroke", c)
    .attr("stroke-width", o.w || 1.4).attr("stroke-linecap", "round").attr("stroke-dasharray", o.dash || null);
  if (o.dot !== false) g.append("circle").attr("cx", x0).attr("cy", y0).attr("r", 1.8).attr("fill", c);
  const ang = (Math.atan2(y1 - c2[1], x1 - c2[0]) * 180) / Math.PI;
  g.append("path").attr("d", "M-4.8,-3.1 L0,0 L-4.8,3.1").attr("transform", `translate(${x1},${y1}) rotate(${ang})`).attr("fill", "none")
    .attr("stroke", c).attr("stroke-width", o.w || 1.4).attr("stroke-linecap", "round").attr("stroke-linejoin", "round");
}
function tile(g, x, y, w, h, c) {
  g.append("rect").attr("x", x).attr("y", y + 1.1).attr("width", w).attr("height", h).attr("rx", 1.8).attr("fill", shade(c, 0.22));
  g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("rx", 1.8).attr("fill", grad([[0, tint(c, 0.6)], [1, c]]));
}
function slab(g, x, y, w, h, d, front, top, side) {
  const dx = d * 0.87, dy = -d * 0.5, e = (s) => s.attr("stroke", "#fff").attr("stroke-width", 0.5).attr("stroke-linejoin", "round");
  e(g.append("path").attr("d", `M${x},${y} L${x + w},${y} L${x + w + dx},${y + dy} L${x + dx},${y + dy} Z`).attr("fill", top));
  e(g.append("path").attr("d", `M${x + w},${y} L${x + w + dx},${y + dy} L${x + w + dx},${y + h + dy} L${x + w},${y + h} Z`).attr("fill", side));
  e(g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("fill", front));
}

const L0 = 114, LG = 17, HL = 5, SY = 52, SH = 70, CX = 274, DX = 494, IW = 120, RY = 182, RH = 34, FX = 190, FW = 148, EX = 352, EW = 128;
const bw = F.box.x1 - F.box.x0, bh = F.box.y1 - F.box.y0, sc = IW / bw, IH = bh * sc, IY = 52;
const C20 = F.cols.find((c) => c.name === "20%"), CF = F.cols.find((c) => c.name === "final");

// Colors for the noisy-motion block (front cells by noise value v in [0, 1], top and side faces).
const NOISE = {
  gray: { name: "neutral gray (current)", cell: (v) => tint(C.dim, 0.18 + 0.62 * v), top: tint(C.dim, 0.25), side: tint(C.dim, 0.55) },
  signed: { name: "signed noise: blue (negative) to magenta (positive)",
    cell: (v) => (v < 0.5 ? d3.interpolateRgb(tint(C.lay2, 0.85), "#f2f2f2")(v * 2) : d3.interpolateRgb("#f2f2f2", tint(C.sec, 0.85))((v - 0.5) * 2)),
    top: tint(C.lay2, 0.18), side: tint(C.sec, 0.4) },
  sunset: { name: "Sunset ramp (the denoising colors of Figure 8)", cell: (v) => ramp(0.05 + 0.9 * v), top: tint(ramp(0.2), 0.55), side: tint(ramp(0.55), 0.7) },
  blue: { name: "light blue, as the layers", cell: (v) => tint(C.lay2, 0.12 + 0.6 * v), top: tint(C.lay2, 0.12), side: tint(C.lay2, 0.42) },
  slate: { name: "slate ink tints", cell: (v) => tint(C.ink, 0.08 + 0.5 * v), top: tint(C.ink, 0.12), side: tint(C.ink, 0.34) },
};

function figure(g, sty, opt = {}) {
  const N = NOISE[opt.noise || "gray"];
  const T = (x, y, s, o) => tx(g, x, y, s, o);
  T(0, 10, "What we compare at each recorded step", { color: C.ink, bold: true, size: 10.5 });
  T(0, 22, "One step of the 50-step MDM at 20% of denoising (step 10 of 50), drawn on a recorded jump");

  // inputs: the noisy motion (an icon: features by frames) and the text
  T(0, 40, "noisy motion #x#_{$t$}", { color: C.ink, bold: true });
  if (sty === "iso" && opt.flat) {  // the input is data, so a flat grid on a card like the other data
    card(g, 0, 45, 88, 30, { r: 5 });
    d3.range(30).forEach((i) => g.append("rect").attr("x", 4 + (i % 10) * 8).attr("y", 49 + Math.floor(i / 10) * 7.4)
      .attr("width", 7.2).attr("height", 6.6).attr("rx", 1).attr("fill", N.cell(q(noise(i)))));
  } else if (sty === "iso") {
    const x0 = 0, y0 = 49, cw = 7.6, ch = 6.4, nc = 10, nr = 4, d = 10, dx = d * 0.87, dy = -d * 0.5, w = nc * cw, h = nr * ch;
    g.append("path").attr("d", `M${x0},${y0} L${x0 + w},${y0} L${x0 + w + dx},${y0 + dy} L${x0 + dx},${y0 + dy} Z`).attr("fill", N.top);
    g.append("path").attr("d", `M${x0 + w},${y0} L${x0 + w + dx},${y0 + dy} L${x0 + w + dx},${y0 + h + dy} L${x0 + w},${y0 + h} Z`).attr("fill", N.side);
    d3.range(nc * nr).forEach((i) => g.append("rect").attr("x", x0 + (i % nc) * cw).attr("y", y0 + Math.floor(i / nc) * ch)
      .attr("width", cw).attr("height", ch).attr("fill", N.cell(q(noise(i)))).attr("stroke", "#fff").attr("stroke-width", 0.6));
  } else if (sty === "soft") {
    card(g, 0, 45, 88, 30, { r: 6 });
    d3.range(36).forEach((i) => g.append("circle").attr("cx", 9 + (i % 12) * 6.4).attr("cy", 52 + Math.floor(i / 12) * 8).attr("r", 2.5)
      .attr("fill", tint(C.dim, 0.2 + 0.65 * q(noise(i)))));
  } else {
    d3.range(30).forEach((i) => tile(g, (i % 10) * 8.6, 46 + Math.floor(i / 10) * 9, 7.4, 7.2, tint(C.dim, 0.2 + 0.65 * q(noise(i)))));
  }
  T(0, 85, "263 features × 120 frames", { size: 8 });
  T(0, 100, "text $u$", { color: C.ink, bold: true });
  card(g, 0, 104, 88, 24, sty === "soft" ? { fill: tint(C.neu, 0.22), shadow: false, r: 6 } : { r: 5 });
  T(7, 114, `“${F.prompt.split(" ").slice(0, 2).join(" ")}`, { size: 8.5, color: C.ink, italic: true });
  T(7, 124.5, `${F.prompt.split(" ").slice(2).join(" ")}”`, { size: 8.5, color: C.ink, italic: true });
  T(0, 139, "CLIP embedding", { size: 8 });

  // the transformer
  T(L0 - 4, 40, "transformer, 8 layers", { color: C.ink, bold: true });
  const lx = (i) => L0 + i * LG;
  let out = [254, SY + SH / 2], lw = 8;
  if (sty === "iso") {
    g.append("path").attr("d", `M${L0 - 6},${SY + SH + 3} L${lx(7) + 8},${SY + SH + 3} L${lx(7) + 18.4},${SY + SH - 3} L${L0 + 4.4},${SY + SH - 3} Z`)
      .attr("fill", "#1c2b4a").attr("opacity", 0.07);
    g.append("line").attr("x1", L0 + 2).attr("x2", 252).attr("y1", SY + SH / 2).attr("y2", SY + SH / 2).attr("stroke", tint(C.lay2, 0.45)).attr("stroke-width", 2.2);
    d3.range(8).forEach((i) => i === HL
      ? slab(g, lx(i), SY, 8, SH, 12, grad([[0, d3.interpolateRgb(C.lay2, C.lay)(0.45)], [1, C.lay]]), tint(C.lay2, 0.6), shade(C.lay, 0.3))
      : slab(g, lx(i), SY, 8, SH, 12, grad([[0, tint(C.lay2, 0.16)], [1, tint(C.lay2, 0.32)]]), tint(C.lay2, 0.1), tint(C.lay2, 0.45)));
    out = [254, SY + SH / 2];
  } else if (sty === "soft") {
    card(g, L0 - 8, SY - 6, 8 * LG + 6, SH + 12, { r: 9, fill: grad([[0, tint(C.lay2, 0.07)], [1, "#ffffff"]]) });
    g.append("line").attr("x1", L0 - 8).attr("x2", lx(7) + 18).attr("y1", SY + SH / 2).attr("y2", SY + SH / 2).attr("stroke", tint(C.lay2, 0.35)).attr("stroke-width", 1.4);
    lw = 10;
    d3.range(8).forEach((i) => {
      if (i === HL) [4.5, 3, 1.5].forEach((k, j) => g.append("rect").attr("x", lx(i) - k).attr("y", SY - k).attr("width", 10 + 2 * k)
        .attr("height", SH + 2 * k).attr("rx", 5 + k).attr("fill", C.lay2).attr("opacity", [0.06, 0.09, 0.13][j]));
      g.append("rect").attr("x", lx(i)).attr("y", SY).attr("width", 10).attr("height", SH).attr("rx", 5)
        .attr("fill", i === HL ? grad([[0, C.lay2], [1, C.lay]]) : grad([[0, tint(C.lay2, 0.34)], [1, tint(C.lay2, 0.14)]]));
    });
    out = [250, SY + SH / 2];
  } else {
    lw = 11;
    d3.range(7).forEach((r) => g.append("line").attr("x1", L0 - 6).attr("x2", lx(7) + 17).attr("y1", 116 - (6 - r) * 10.4)
      .attr("y2", 116 - (6 - r) * 10.4).attr("stroke", tint(C.lay2, r === 6 ? 0 : 0.28)).attr("stroke-width", 0.8));
    d3.range(8).forEach((i) => d3.range(7).forEach((r) => {
      const v = q(noise(i * 7 + r + 50));
      const c = r === 6 ? tint(C.dim, 0.3 + 0.25 * v) : i === HL ? d3.interpolateRgb(C.lay2, C.lay)(0.35 + 0.65 * v) : tint(C.lay2, 0.18 + 0.36 * v);
      tile(g, lx(i), 112 - (6 - r) * 10.4, 11, 8, c);
    }));
    out = [250, SY + SH / 2];
  }
  d3.range(8).forEach((i) => T(lx(i) + lw / 2, SY + SH + 13, `${i}`, { anchor: "middle", size: 8, color: i === HL ? C.lay : C.muted, bold: i === HL }));
  T(lx(HL) + lw / 2, SY + SH + 25, "layer $ℓ$", { anchor: "middle", size: 8.5, color: C.lay, bold: true });
  link(g, [92, 60], [L0 - 10, 60], C.dim, { dot: false });
  link(g, [92, 116], [L0 - 10, 116], C.dim, { dot: false });

  // the estimate and the final motion as bodies on cards, center of mass red while airborne
  function body(c, ox) {
    card(g, ox - 4, IY - 4, IW + 8, IH + 8, { r: 6 });
    g.append("image").attr("href", c.img).attr("x", ox).attr("y", IY).attr("width", IW).attr("height", IH);
    const X = (p) => ox + (p[0] - F.box.x0) * sc, Y = (p) => IY + (F.box.y1 - p[1]) * sc;
    g.append("path").attr("d", d3.line().defined((p, i) => c.air[i]).x(X).y(Y)(c.com)).attr("fill", "none").attr("stroke", C.mark).attr("stroke-width", 1.6);
  }
  T(CX - 4, 40, "clean-motion estimate #x̂#_{0}", { color: C.est, bold: true });
  body(C20, CX);
  link(g, out, [CX - 8, SY + SH / 2], C.est, { dot: false });
  T(CX - 4, IY + IH + 16, "joints and every label follow", { size: 8 });
  link(g, [CX + IW + 8, IY + IH / 2], [DX - 8, IY + IH / 2], C.dim, { dash: "3 3", dot: false });
  T((CX + IW + DX) / 2, IY + IH / 2 - 7, "sampler continues", { anchor: "middle", size: 8 });
  T((CX + IW + DX) / 2, IY + IH / 2 + 13, "40 more steps", { anchor: "middle", size: 8 });
  T(DX - 4, 40, "final motion #x#_{0}", { color: C.ink, bold: true });
  body(CF, DX);
  T(DX - 4, IY + IH + 16, `fall rate ${CF.g.toFixed(1)} m/s²`, { size: 8.5, color: C.mark, bold: true });

  // readouts
  [[FX, FW, C.lay, "estimate + layer", "$f$(#x̂#_{0}, #a#_{$ℓ$}) → $ŷ$"], [EX, EW, C.est, "estimate", "$f$(#x̂#_{0}) → $ŷ$"]].forEach(([x, w, c, n, f]) => {
    if (sty === "iso") { card(g, x, RY, w, RH); g.append("rect").attr("x", x + 5).attr("y", RY + 6).attr("width", 3).attr("height", RH - 12).attr("rx", 1.5).attr("fill", c); }
    else if (sty === "soft") card(g, x, RY, w, RH, { fill: grad([[0, tint(c, 0.13)], [1, tint(c, 0.04)]]), stroke: tint(c, 0.45), sw: 0.8 });
    else { card(g, x, RY, w, RH, { fill: c }); g.append("rect").attr("x", x).attr("y", RY + 3).attr("width", w).attr("height", RH - 3).attr("rx", 5).attr("fill", "#fff"); }
    const ix = sty === "iso" ? x + 15 : x + 9;
    T(ix, RY + 14, n, { color: c, bold: true });
    T(ix, RY + 27, f, { color: C.ink });
  });
  const hx = lx(HL) + lw / 2;
  link(g, [hx, SY + SH + 31], [FX + 34, RY - 3], C.lay, { mode: "v", w: 1.5 });
  T(hx - 6, RY - 13, "activations #a#_{$ℓ$}", { anchor: "end", size: 8.5, color: C.lay, bold: true });
  const eb = IY + IH + 22;
  link(g, [CX + 28, eb], [FX + FW - 28, RY - 3], C.est, { mode: "v", w: 1.5 });
  link(g, [CX + 92, eb], [EX + 52, RY - 3], C.est, { mode: "v", w: 1.5 });
  link(g, [DX + IW / 2, eb], [DX + IW / 2, RY - 4], C.mark, { mode: "v", w: 1.4, dash: "3 2.5" });
  T(DX - 4, RY + 10, "target $y$", { color: C.mark, bold: true });
  T(DX - 4, RY + 22, "the final motion's label,", { size: 8 });
  T(DX - 4, RY + 32, "one per frame", { size: 8 });
  T(FX, RY + RH + 14, "both tuned (ridge or MLP) and scored on held-out prompts", { size: 8 });
  T(FX, RY + RH + 28, "gain from the layer = $R$²(estimate + layer) − $R$²(estimate)", { color: C.ink });

  // what differs between the two models
  const TY = RY + 6, TC = [0, 46, 112];
  [["", "MDM", "MotionDiffuse"], ["outputs", "estimate #x̂#_{0}", "noise #ε̂#"], ["text", "input token", "every layer"],
   ["guidance", "scale 2.5", "none"], ["steps", "50 or 1000", "1000"]].forEach((row, r) => row.forEach((s, j) => {
    if (s) T(TC[j], TY + r * 12.5, s, { size: 8.5, color: r === 0 || j > 0 ? C.ink : C.muted, bold: r === 0 });
  }));
  g.append("line").attr("x1", 0).attr("x2", 176).attr("y1", TY + 4).attr("y2", TY + 4).attr("stroke", C.axis).attr("stroke-width", 0.6);
}

// hats need the final glyph positions, so draw them once the fonts are loaded (pages call this last)
const finish = () => Promise.all(["9px Literata", "italic 9px Literata", "650 9px Literata"].map((f) => document.fonts.load(f))).then(drawHats);
