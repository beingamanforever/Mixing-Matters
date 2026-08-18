/* Mixing Matters project page.
   Charts are hand-rolled SVG so the page stays dependency-free and works from
   a plain static host. Three renderers cover every figure: a line/dot chart on
   shared axes, grouped horizontal bars anchored at zero, and a labelled scatter. */

const NS = "http://www.w3.org/2000/svg";

const MIXER = {
  attention: "var(--attention)",
  "state-space": "var(--state-space)",
  hybrid: "var(--hybrid)",
};

const INK = "var(--ink)";

/* ---------- small helpers ---------- */

const $ = (selector, scope = document) => scope.querySelector(selector);

function make(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  if (parent) parent.appendChild(node);
  return node;
}

function html(tag, attrs = {}, parent) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  if (parent) parent.appendChild(node);
  return node;
}

const pp = (value, digits = 2) => {
  const magnitude = Math.abs(value * 100).toFixed(digits);
  return Number(magnitude) === 0 ? magnitude : `${value > 0 ? "+" : "−"}${magnitude}`;
};
const acc = (value) => value.toFixed(3);

/** Axis ticks in percentage points, at just enough precision for the step size. */
const ppTick = (value, step) => {
  const points = step * 100;
  return pp(value, points >= 1 && Number.isInteger(points) ? 0 : points >= 1 ? 1 : 2);
};
const pval = (value) => (value < 0.0001 ? "<0.0001" : value.toFixed(4));
const significant = (effect) => effect.p < 0.05;

/* ---------- scales ---------- */

function linear(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const scale = (value) => r0 + ((value - d0) / span) * (r1 - r0);
  scale.invert = (value) => d0 + ((value - r0) / (r1 - r0)) * span;
  scale.domain = domain;
  return scale;
}

function band(count, range, padding = 0.5) {
  const step = (range[1] - range[0]) / (count - 1 + padding * 2);
  const scale = (index) => range[0] + step * (padding + index);
  scale.step = step;
  return scale;
}

/** Round a domain outward to a readable tick sequence of about `target` steps. */
function ticks([lo, hi], target = 5) {
  const span = hi - lo || 1;
  const raw = span / target;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].find((factor) => factor * magnitude >= raw * 0.999) * magnitude;
  const start = Math.ceil(lo / step - 1e-9) * step;
  const out = [];
  for (let value = start; value <= hi + step * 1e-9; value += step) {
    out.push(Math.abs(value) < step * 1e-9 ? 0 : Number(value.toFixed(10)));
  }
  return { values: out, step };
}

function pad([lo, hi], fraction = 0.08) {
  const span = hi - lo || Math.abs(hi) || 1;
  return [lo - span * fraction, hi + span * fraction];
}

/* ---------- shared chart chrome ---------- */

function frame(container, height, minWidth = 320) {
  container.textContent = "";
  // Below the floor the chart scrolls inside its own container rather than
  // collapsing its labels; `.chart` carries the horizontal scroll.
  const width = Math.max(container.clientWidth || 640, minWidth);
  const svg = make("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    role: "img",
  }, container);
  return { svg, width, height };
}

function yAxis(svg, scale, box, { label, format = acc, count = 5 }) {
  const { values, step } = ticks(scale.domain, count);
  for (const value of values) {
    const y = scale(value);
    make("line", {
      class: "grid-line",
      x1: box.left,
      x2: box.right,
      y1: y,
      y2: y,
    }, svg);
    make("text", {
      class: "tick-text",
      x: box.left - 9,
      y: y + 3.5,
      "text-anchor": "end",
    }, svg).textContent = format(value, step);
  }
  if (label) {
    const text = make("text", {
      class: "axis-title",
      transform: `rotate(-90 ${14} ${(box.top + box.bottom) / 2})`,
      x: 14,
      y: (box.top + box.bottom) / 2,
      "text-anchor": "middle",
    }, svg);
    text.textContent = label;
  }
}

function xAxis(svg, box, entries, label, rotate = false) {
  make("line", {
    class: "axis-line",
    x1: box.left,
    x2: box.right,
    y1: box.bottom,
    y2: box.bottom,
  }, svg);
  for (const entry of entries) {
    const y = box.bottom + 17;
    const text = make("text", {
      class: "tick-text",
      x: entry.x,
      y,
      "text-anchor": rotate ? "end" : "middle",
      transform: rotate ? `rotate(-32 ${entry.x} ${y})` : null,
    }, svg);
    text.textContent = entry.label;
  }
  if (label) {
    const text = make("text", {
      class: "axis-title",
      x: (box.left + box.right) / 2,
      y: box.bottom + (rotate ? 92 : 38),
      "text-anchor": "middle",
    }, svg);
    text.textContent = label;
  }
}

/** Squeeze an over-long label rather than let it collide with the plot. */
function fitText(node, maxWidth) {
  if (node.getComputedTextLength && node.getComputedTextLength() > maxWidth) {
    node.setAttribute("textLength", maxWidth);
    node.setAttribute("lengthAdjust", "spacingAndGlyphs");
  }
}

/** Nudge stacked end-labels apart so a crowded panel stays readable. */
function spread(items, minGap = 13) {
  const sorted = [...items].sort((a, b) => a.y - b.y);
  for (let i = 1; i < sorted.length; i += 1) {
    const gap = sorted[i].y - sorted[i - 1].y;
    if (gap < minGap) sorted[i].y = sorted[i - 1].y + minGap;
  }
  return items;
}

/* ---------- tooltip ---------- */

function tooltipFor(container) {
  const node = html("div", { class: "tooltip" }, container);
  return {
    show(x, y, content) {
      node.innerHTML = content;
      node.dataset.open = "true";
      const width = node.offsetWidth;
      const left = Math.min(Math.max(x - width / 2, 4), container.clientWidth - width - 4);
      node.style.left = `${left}px`;
      node.style.top = `${Math.max(y - node.offsetHeight - 12, 4)}px`;
    },
    hide() {
      node.dataset.open = "false";
    },
  };
}

/* ---------- renderer: line and categorical dot charts ---------- */

/**
 * spec = {
 *   height, x: {kind, values, label, format}, y: {label, format, domain, zero},
 *   series: [{label, color, dash, hollow, band, connect, points:[{x,y,lo,hi}]}],
 *   directLabels, rule: {y, label}
 * }
 */
function renderLine(container, spec) {
  const height = spec.height || 300;
  const { svg, width } = frame(container, height);
  const directLabels = spec.directLabels && width >= 520;
  const box = {
    left: 52,
    right: width - (directLabels ? Math.min(150, width * 0.24) : 16),
    top: 12,
    bottom: height - (spec.x.rotate ? 100 : spec.x.label ? 46 : 28),
  };

  const allY = spec.series.flatMap((series) =>
    series.points.flatMap((point) => [point.y, point.lo ?? point.y, point.hi ?? point.y])
  );
  const domain = spec.y.domain || pad([Math.min(...allY), Math.max(...allY)]);
  if (spec.y.zero && domain[0] > 0) domain[0] = 0;
  const y = linear(domain, [box.bottom, box.top]);

  const xs = spec.x.values;
  const x =
    spec.x.kind === "band"
      ? band(xs.length, [box.left, box.right])
      : linear(pad([Math.min(...xs), Math.max(...xs)], 0.05), [box.left, box.right]);

  const at = (value, index) => (spec.x.kind === "band" ? x(index) : x(value));

  yAxis(svg, y, box, spec.y);
  xAxis(
    svg,
    box,
    xs.map((value, index) => ({ x: at(value, index), label: spec.x.format ? spec.x.format(value) : String(value) })),
    spec.x.label,
    spec.x.rotate
  );

  if (spec.y.zero && domain[0] < 0 && domain[1] > 0) {
    make("line", { class: "zero-line", x1: box.left, x2: box.right, y1: y(0), y2: y(0) }, svg);
  }
  if (spec.rule) {
    make("line", {
      class: "zero-line",
      x1: box.left,
      x2: box.right,
      y1: y(spec.rule.y),
      y2: y(spec.rule.y),
      "stroke-dasharray": "4 4",
    }, svg);
    make("text", {
      class: "point-label",
      x: box.left + 6,
      y: y(spec.rule.y) - 6,
    }, svg).textContent = spec.rule.label;
  }

  // Confidence bands sit behind every mark.
  for (const series of spec.series) {
    if (!series.band) continue;
    const upper = series.points.map((point, index) => `${at(point.x, index)},${y(point.hi)}`);
    const lower = series.points.map((point, index) => `${at(point.x, index)},${y(point.lo)}`).reverse();
    make("polygon", {
      points: [...upper, ...lower].join(" "),
      fill: series.color,
      "fill-opacity": 0.11,
    }, svg);
  }

  // A dumbbell reads the within-category difference; dodged dots read the
  // between-category trend. Both are band charts, so the caller picks.
  if (spec.connectors) {
    for (const [index] of spec.series[0].points.entries()) {
      const values = spec.series.map((series) => series.points[index].y);
      make("line", {
        x1: at(spec.series[0].points[index].x, index),
        x2: at(spec.series[0].points[index].x, index),
        y1: y(Math.min(...values)),
        y2: y(Math.max(...values)),
        stroke: "var(--rule)",
        "stroke-width": 2,
      }, svg);
    }
  }

  const labels = [];
  spec.series.forEach((series, order) => {
    const dodge = spec.dodge !== false && spec.series.length > 1 && spec.x.kind === "band";
    const offset = dodge && !series.connect ? (order - (spec.series.length - 1) / 2) * 9 : 0;
    if (series.connect !== false) {
      const path = series.points
        .map((point, index) => `${index ? "L" : "M"}${at(point.x, index) + offset},${y(point.y)}`)
        .join(" ");
      make("path", {
        d: path,
        fill: "none",
        stroke: series.color,
        "stroke-width": 2,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
        "stroke-dasharray": series.dash ? "6 4" : null,
      }, svg);
    }
    for (const [index, point] of series.points.entries()) {
      const cx = at(point.x, index) + offset;
      if (point.lo !== undefined && !series.band) {
        make("line", {
          x1: cx,
          x2: cx,
          y1: y(point.lo),
          y2: y(point.hi),
          stroke: series.color,
          "stroke-width": 1.5,
          "stroke-linecap": "round",
        }, svg);
      }
      make("circle", {
        cx,
        cy: y(point.y),
        r: 4.5,
        fill: series.hollow ? "var(--surface)" : series.color,
        stroke: series.color,
        "stroke-width": 2,
      }, svg);
    }
    if (directLabels) {
      const last = series.points[series.points.length - 1];
      labels.push({
        y: y(last.y),
        x: at(last.x, series.points.length - 1) + offset + 11,
        text: series.label,
        color: series.color,
      });
    }
  });

  for (const label of spread(labels)) {
    const text = make("text", {
      class: "series-label",
      x: label.x,
      y: label.y + 4,
      fill: label.color,
    }, svg);
    text.textContent = label.text;
  }

  attachCrosshair(container, svg, spec, { box, at, y, xs });
}

/** One crosshair per x slot, listing every series at that slot. */
function attachCrosshair(container, svg, spec, { box, at, y, xs }) {
  const tip = tooltipFor(container);
  const line = make("line", {
    class: "zero-line",
    y1: box.top,
    y2: box.bottom,
    stroke: "#c6c9cc",
    opacity: 0,
  }, svg);

  const positions = xs.map((value, index) => at(value, index));
  const overlay = make("rect", {
    x: box.left,
    y: box.top,
    width: Math.max(box.right - box.left, 1),
    height: Math.max(box.bottom - box.top, 1),
    fill: "transparent",
  }, svg);

  const format = spec.y.tipFormat || spec.y.format || acc;
  const move = (event) => {
    const rect = svg.getBoundingClientRect();
    const scale = rect.width / svg.viewBox.baseVal.width;
    const px = (event.clientX - rect.left) / scale;
    let nearest = 0;
    positions.forEach((value, index) => {
      if (Math.abs(value - px) < Math.abs(positions[nearest] - px)) nearest = index;
    });
    line.setAttribute("x1", positions[nearest]);
    line.setAttribute("x2", positions[nearest]);
    line.setAttribute("opacity", 0.6);
    const heading = spec.x.tipLabel ? spec.x.tipLabel(xs[nearest]) : `Position ${xs[nearest]}`;
    const rows = spec.series
      .map((series) => {
        const point = series.points[nearest];
        if (!point) return "";
        const range = point.lo === undefined ? "" : ` <span class="v">[${format(point.lo)}, ${format(point.hi)}]</span>`;
        return `<div class="row" style="color:${series.color}"><i></i><span style="color:#fff">${series.label}</span><span class="v">${format(point.y)}</span>${range}</div>`;
      })
      .join("");
    tip.show(positions[nearest] * scale, y(spec.series[0].points[nearest]?.y ?? 0) * scale, `<b>${heading}</b>${rows}`);
  };

  overlay.addEventListener("pointermove", move);
  overlay.addEventListener("pointerleave", () => {
    tip.hide();
    line.setAttribute("opacity", 0);
  });
}

/* ---------- renderer: horizontal bars anchored at zero ---------- */

/** A bar that is square at the baseline and rounded at the value end. */
function barPath(x0, x1, top, height, radius = 4) {
  const bottom = top + height;
  const r = Math.min(radius, Math.abs(x1 - x0));
  const sign = x1 >= x0 ? 1 : -1;
  const end = x1 - r * sign;
  const sweep = sign > 0 ? 1 : 0;
  return [
    `M${x0},${top}`,
    `H${end}`,
    `A${r},${r} 0 0 ${sweep} ${x1},${top + r}`,
    `V${bottom - r}`,
    `A${r},${r} 0 0 ${sweep} ${end},${bottom}`,
    `H${x0}`,
    "Z",
  ].join(" ");
}

/**
 * spec = {
 *   groups: [{label, bars: [{label, value, lo, hi, color, hollow, note}]}],
 *   xLabel, labelWidth, tickFormat, markFormat, valueLabel
 * }
 */
function renderBars(container, spec) {
  const barHeight = spec.barHeight || 15;
  const gap = 2;
  const groupGap = 16;
  const groupHeight = (row) => row.bars.length * barHeight + (row.bars.length - 1) * gap;
  const plotHeight = spec.groups.reduce((total, row) => total + groupHeight(row) + groupGap, 0);
  const height = plotHeight + 44;
  const { svg, width } = frame(container, height, 470);
  const labelWidth = Math.min(spec.labelWidth || 210, width * 0.4);
  const box = { left: labelWidth, right: width - 54, top: 8, bottom: height - 36 };
  const tick = spec.tickFormat || ppTick;
  const mark = spec.markFormat || ((value) => pp(value));

  const values = spec.groups.flatMap((row) => row.bars.flatMap((bar) => [bar.lo, bar.hi, bar.value, 0]));
  const x = linear(pad([Math.min(...values), Math.max(...values)], 0.1), [box.left, box.right]);

  const { values: tickValues, step } = ticks(x.domain, 5);
  for (const value of tickValues) {
    make("line", { class: "grid-line", x1: x(value), x2: x(value), y1: box.top, y2: box.bottom }, svg);
    make("text", {
      class: "tick-text",
      x: x(value),
      y: box.bottom + 18,
      "text-anchor": "middle",
    }, svg).textContent = tick(value, step);
  }

  const tip = tooltipFor(container);
  let cursor = box.top + groupGap / 2;
  for (const row of spec.groups) {
    const span = groupHeight(row);
    const label = make("text", {
      class: "row-label",
      x: 0,
      y: cursor + span / 2 + 4,
    }, svg);
    label.textContent = row.label;
    fitText(label, labelWidth - 14);

    row.bars.forEach((bar, index) => {
      const top = cursor + index * (barHeight + gap);
      make("path", {
        d: barPath(x(0), x(bar.value), top, barHeight),
        fill: bar.color,
        "fill-opacity": bar.hollow ? 0.28 : 1,
      }, svg);
      if (bar.lo !== undefined) {
        const middle = top + barHeight / 2;
        const cap = 4;
        make("path", {
          d:
            `M${x(bar.lo)},${middle - cap} V${middle + cap} M${x(bar.lo)},${middle} ` +
            `H${x(bar.hi)} M${x(bar.hi)},${middle - cap} V${middle + cap}`,
          fill: "none",
          stroke: "var(--ink)",
          "stroke-opacity": 0.55,
          "stroke-width": 1.5,
        }, svg);
      }
      const value = make("text", {
        class: "point-label",
        x: box.right + 8,
        y: top + barHeight / 2 + 4,
      }, svg);
      value.textContent = mark(bar.value);

      const target = make("rect", {
        x: box.left,
        y: top - gap / 2,
        width: Math.max(box.right - box.left, 1),
        height: barHeight + gap,
        fill: "transparent",
      }, svg);
      target.addEventListener("pointerenter", () => {
        const rect = svg.getBoundingClientRect();
        const scale = rect.width / svg.viewBox.baseVal.width;
        tip.show(
          x(bar.value) * scale,
          (top + barHeight / 2) * scale,
          `<b>${bar.label || row.label}</b>` +
            `<div class="row"><span>${spec.valueLabel || "Estimate"}</span>` +
            `<span class="v">${mark(bar.value)}</span></div>` +
            `<div class="row"><span>95% interval</span>` +
            `<span class="v">[${mark(bar.lo)}, ${mark(bar.hi)}]</span></div>` +
            (bar.note ? `<div class="row"><span>Holm p</span><span class="v">${bar.note}</span></div>` : "")
        );
      });
      target.addEventListener("pointerleave", () => tip.hide());
    });

    cursor += span + groupGap;
  }

  make("line", { class: "zero-line", x1: x(0), x2: x(0), y1: box.top, y2: box.bottom }, svg);
  const title = make("text", {
    class: "axis-title",
    x: (box.left + box.right) / 2,
    y: height - 6,
    "text-anchor": "middle",
  }, svg);
  title.textContent = spec.xLabel;
}

/* ---------- renderer: labelled scatter ---------- */

/** spec = { points:[{x, y, lo, hi, label, color}], x:{label, format}, y:{label} } */
function renderScatter(container, spec) {
  const height = spec.height || 300;
  const { svg, width } = frame(container, height);
  const box = { left: 56, right: width - 24, top: 26, bottom: height - 46 };

  const xDomain = pad([Math.min(...spec.points.map((p) => p.x)), Math.max(...spec.points.map((p) => p.x))], 0.14);
  const yValues = spec.points.flatMap((p) => [p.lo ?? p.y, p.hi ?? p.y, 0]);
  const x = linear(xDomain, [box.left, box.right]);
  const y = linear(pad([Math.min(...yValues), Math.max(...yValues)], 0.14), [box.bottom, box.top]);

  yAxis(svg, y, box, { label: spec.y.label, format: ppTick });
  const { values: xTicks } = ticks(xDomain, 5);
  xAxis(
    svg,
    box,
    xTicks.map((value) => ({ x: x(value), label: spec.x.format ? spec.x.format(value) : value.toFixed(2) })),
    spec.x.label
  );
  make("line", { class: "zero-line", x1: box.left, x2: box.right, y1: y(0), y2: y(0) }, svg);

  const labels = [];
  for (const point of spec.points) {
    if (point.lo !== undefined) {
      make("line", {
        x1: x(point.x),
        x2: x(point.x),
        y1: y(point.lo),
        y2: y(point.hi),
        stroke: point.color,
        "stroke-width": 1.5,
        "stroke-linecap": "round",
      }, svg);
    }
    make("circle", {
      cx: x(point.x),
      cy: y(point.y),
      r: 4.5,
      fill: point.color,
      stroke: "var(--surface)",
      "stroke-width": 2,
    }, svg);
    // Sit the label above the interval so it never lands on a neighbour's bar.
    labels.push({ x: x(point.x), y: y(point.hi ?? point.y) - 9, text: point.label });
  }
  for (const label of labels) {
    const text = make("text", {
      class: "point-label",
      x: Math.min(Math.max(label.x, box.left + 12), box.right - 12),
      y: label.y,
      "text-anchor": "middle",
    }, svg);
    text.textContent = label.text;
  }
}

/* ---------- figure scaffolding ---------- */

/* Charts render at their container's true pixel width so text stays at its
   intended size. Observing each container covers the first paint, grid reflow,
   panel switching, and window resizing with one mechanism. */
const charts = new Map();

const observer = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const width = Math.round(entry.contentRect.width);
    const chart = charts.get(entry.target);
    if (!chart || width < 1 || width === chart.width) continue;
    chart.width = width;
    chart.draw(entry.target);
  }
});

const pending = [];

/** Register a chart. The first draw waits for `flush`, once the DOM is settled. */
function mount(container, draw) {
  charts.set(container, { draw, width: 0 });
  pending.push(container);
  observer.observe(container);
}

/**
 * Draw every chart registered since the last call.
 *
 * Charts cannot draw as they mount: a grid column is full width until its
 * sibling arrives, so an eager draw measures the wrong box. Deferring to the
 * observer instead is not an option either, because a hidden document (a page
 * opened in a background tab) receives no observer callbacks and would show
 * nothing at all. Drawing explicitly, after the section is built, is correct in
 * both cases.
 */
function flush() {
  for (const container of pending.splice(0)) {
    const chart = charts.get(container);
    chart.width = Math.round(container.getBoundingClientRect().width);
    chart.draw(container);
  }
}

/** Redraw any chart whose container has since been laid out at another width. */
function resize() {
  for (const [container, chart] of charts) {
    if (!container.isConnected) {
      charts.delete(container);
      continue;
    }
    const width = Math.round(container.getBoundingClientRect().width);
    if (width < 1 || width === chart.width) continue;
    chart.width = width;
    chart.draw(container);
  }
}

// A hidden document reports zero-width containers and gets no observer
// callbacks, so charts mounted before the first paint need re-measuring once
// the page actually becomes visible.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) resize();
});

function figure(parent, { id, title, caption, controls }) {
  const node = html("figure", { class: "figure", id }, parent);
  const head = html("div", { class: "figure-head" }, node);
  html("h3", { text: title }, head);
  if (controls) head.appendChild(controls);
  const body = html("div", {}, node);
  if (caption) html("figcaption", { html: caption }, node);
  return body;
}

function legend(parent, series) {
  const node = html("div", { class: "legend" }, parent);
  for (const item of series) {
    const span = html("span", { style: `color:${item.color}` }, node);
    // The swatch mirrors the mark: a rule for connected series, a dot otherwise.
    const shape = item.connect === false ? "mark" : item.dash ? "dashed" : "";
    const fill = item.hollow ? " hollow" : item.faded ? " faded" : "";
    html("i", { class: `${shape}${fill}` }, span);
    html("span", { text: item.label, style: "color:var(--ink-soft)" }, span);
  }
  return node;
}

function toggle(labels, onChange) {
  const node = html("div", { class: "toggle", role: "group" });
  const buttons = labels.map((label, index) => {
    const button = html("button", { type: "button", text: label, "aria-pressed": index === 0 }, node);
    button.addEventListener("click", () => {
      buttons.forEach((other, position) => other.setAttribute("aria-pressed", position === index));
      onChange(index);
    });
    return button;
  });
  return node;
}

function table(parent, columns, rows, caption) {
  const wrap = html("div", { class: "table-wrap" }, parent);
  const node = html("table", {}, wrap);
  if (caption) html("caption", { text: caption }, node);
  const head = html("tr", {}, html("thead", {}, node));
  for (const column of columns) {
    html("th", { text: column.label, class: column.numeric ? "n" : "" }, head);
  }
  const body = html("tbody", {}, node);
  for (const row of rows) {
    const tr = html("tr", {}, body);
    columns.forEach((column, index) => {
      html("td", { html: row[index], class: column.numeric ? "n" : "" }, tr);
    });
  }
  return wrap;
}

/** Chart / table switch. Satisfies the relief rule for the low-contrast series hue. */
function withTable(parent, spec) {
  const chartBox = html("div", {}, parent);
  const tableBox = html("div", { hidden: "" }, parent);
  spec.buildChart(chartBox);
  spec.buildTable(tableBox);
  return toggle(["Chart", "Table"], (index) => {
    chartBox.hidden = index === 1;
    tableBox.hidden = index === 0;
  });
}

/* ---------- page sections ---------- */

let DATA = null;

const styleFor = (models, key) => {
  const mixer = DATA.models[key].mixer;
  const sameMixer = models.filter((other) => DATA.models[other].mixer === mixer);
  const rank = sameMixer.indexOf(key);
  return { color: MIXER[mixer], dash: rank > 0, hollow: rank > 0 };
};

function effectCell(effect) {
  return `${pp(effect.estimate)} <span style="color:var(--ink-mute)">[${pp(effect.ci[0])}, ${pp(effect.ci[1])}]</span>`;
}

function buildHeadline() {
  const parent = $("#headline-stats");
  const panel = DATA.panels.find((item) => item.id === "phase2");
  for (const key of panel.models) {
    const effect = panel.edges[key].primacy;
    const model = DATA.models[key];
    const stat = html("div", { class: "stat", "data-mixer": model.mixer }, parent);
    html("div", { class: "value", text: pp(effect.estimate) }, stat);
    html("div", { class: "label", text: model.label }, stat);
    html("div", {
      class: "detail",
      text: `${model.mixer} mixer · 95% CI [${pp(effect.ci[0])}, ${pp(effect.ci[1])}] · Holm p ${pval(effect.p)}`,
    }, stat);
  }
}

function buildPanels() {
  const parent = $("#panels");
  let current = 0;

  const filters = html("div", { class: "filters" }, parent);
  const body = html("div", {}, parent);

  const draw = () => {
    body.textContent = "";
    // Apply the figure padding before mounting, so the chart is measured at the
    // width it will actually occupy rather than the full band width.
    body.className = "figure";
    const panel = DATA.panels[current];
    const series = panel.models.map((key) => {
      const style = styleFor(panel.models, key);
      return {
        key,
        label: DATA.models[key].label,
        color: style.color,
        dash: style.dash,
        hollow: false,
        band: true,
        points: panel.curves[key].map((point) => ({
          x: point.position,
          y: point.accuracy,
          lo: point.ci[0],
          hi: point.ci[1],
        })),
      };
    });

    const chartControls = withTable(html("div", {}, body), {
      buildChart(container) {
        legend(container, series);
        const chart = html("div", { class: "chart" }, container);
        mount(chart, (node) =>
          renderLine(node, {
            height: 340,
            directLabels: true,
            x: {
              kind: "linear",
              values: series[0].points.map((point) => point.x),
              label: "Position of the answer-bearing passage",
            },
            y: { label: "Accuracy", format: (value) => value.toFixed(2), tipFormat: acc },
            series,
          })
        );
      },
      buildTable(container) {
        const columns = [
          { label: "Model" },
          { label: "Primacy", numeric: true },
          { label: "Recency", numeric: true },
          { label: "Holm p (primacy)", numeric: true },
          { label: "Floor", numeric: true },
          { label: "Ceiling", numeric: true },
        ];
        const rows = panel.models.map((key) => {
          const style = styleFor(panel.models, key);
          const edge = panel.edges[key];
          return [
            `<span class="dot${style.dash ? " hollow" : ""}" style="background:${style.color};color:${style.color}"></span>${DATA.models[key].label}`,
            effectCell(edge.primacy),
            effectCell(edge.recency),
            pval(edge.primacy.p),
            acc(panel.floor_ceiling[key].floor_accuracy),
            acc(panel.floor_ceiling[key].ceiling_accuracy),
          ];
        });
        table(container, columns, rows, "Edge effects in percentage points, with 95% bootstrap intervals.");
      },
    });

    const head = html("div", { class: "figure-head" });
    html("h3", { text: panel.title }, head);
    head.appendChild(chartControls);
    body.insertBefore(head, body.firstChild);
    html("figcaption", {
      html: `${panel.caption} Shaded ribbons are 95% bootstrap intervals over 800 question bundles. Primacy is positions 1-2 minus 5-6; recency is positions 9-10 minus 5-6.`,
    }, body);
    flush();
  };

  DATA.panels.forEach((panel, index) => {
    const button = html("button", { type: "button", text: panel.title, "aria-pressed": index === 0 }, filters);
    button.addEventListener("click", () => {
      current = index;
      [...filters.children].forEach((other, position) => other.setAttribute("aria-pressed", position === index));
      draw();
    });
  });
  draw();
}

function buildContrasts() {
  const parent = $("#contrasts");
  const kinds = [
    { key: "all", label: "All contrasts" },
    { key: "architecture", label: "Architecture" },
    { key: "attention", label: "Attention" },
    { key: "scale", label: "Scale" },
    { key: "corpus", label: "Corpus" },
    { key: "production", label: "Production" },
  ];
  let kind = "all";
  let measure = "primacy";

  const filters = html("div", { class: "filters" }, parent);
  const node = html("figure", { class: "figure" }, parent);
  const head = html("div", { class: "figure-head" }, node);
  const title = html("h3", { text: "Paired primacy differences" }, head);
  const body = html("div", {}, node);
  html("figcaption", {
    html:
      "Each bar is a paired contrast on the same question bundles, in percentage points. Solid bars clear Holm " +
      "correction at p &lt; 0.05; faded bars do not. The rule through each bar is its 95% bootstrap interval.",
  }, node);

  head.appendChild(
    toggle(["Primacy", "Recency"], (index) => {
      measure = index === 0 ? "primacy" : "recency";
      title.textContent = index === 0 ? "Paired primacy differences" : "Paired recency differences";
      draw();
    })
  );

  function draw() {
    body.textContent = "";
    const groups = DATA.contrasts
      .filter((row) => kind === "all" || row.kind === kind)
      .map((row) => {
        const effect = row[measure];
        return {
          label: row.label,
          bars: [
            {
              value: effect.estimate,
              lo: effect.ci[0],
              hi: effect.ci[1],
              note: pval(effect.p),
              color: "var(--attention)",
              hollow: !significant(effect),
            },
          ],
        };
      });
    const chart = html("div", { class: "chart" }, body);
    mount(chart, (container) =>
      renderBars(container, {
        groups,
        barHeight: 17,
        labelWidth: 260,
        xLabel: `Paired ${measure} difference (percentage points)`,
      })
    );
    flush();
  }

  kinds.forEach((entry, index) => {
    const button = html("button", { type: "button", text: entry.label, "aria-pressed": index === 0 }, filters);
    button.addEventListener("click", () => {
      kind = entry.key;
      [...filters.children].forEach((other, position) => other.setAttribute("aria-pressed", position === index));
      draw();
    });
  });
  draw();
}

function buildScale() {
  const parent = $("#scale");
  const grid = html("div", { class: "grid-2" }, parent);

  const attention = {
    label: "Attention (Pythia)",
    color: MIXER.attention,
    points: DATA.scale.map((entry) => ({
      x: entry.params_millions,
      y: entry.attention.primacy.estimate,
      lo: entry.attention.primacy.ci[0],
      hi: entry.attention.primacy.ci[1],
    })),
  };
  const stateSpace = {
    label: "State-space (Mamba)",
    color: MIXER["state-space"],
    points: DATA.scale.map((entry) => ({
      x: entry.params_millions,
      y: entry.state_space.primacy.estimate,
      lo: entry.state_space.primacy.ci[0],
      hi: entry.state_space.primacy.ci[1],
    })),
  };
  const difference = {
    label: "Attention minus state-space",
    color: INK,
    points: DATA.scale.map((entry) => ({
      x: entry.params_millions,
      y: entry.difference.estimate,
      lo: entry.difference.ci[0],
      hi: entry.difference.ci[1],
    })),
  };

  const xSpec = {
    kind: "band",
    values: DATA.scale.map((entry) => entry.params_millions),
    label: "Mean parameters of the pair (millions)",
    format: (value) => (value >= 1000 ? `${(value / 1000).toFixed(1)}B` : `${Math.round(value)}M`),
    tipLabel: (value) => `${Math.round(value)}M parameter pair`,
  };
  const ySpec = { label: "Primacy effect (pp)", format: ppTick, tipFormat: (value) => pp(value), zero: true };

  const left = figure(grid, {
    title: "Primacy by scale, per family",
    caption: "Five approximate size pairs, each run on the same 800 questions. Bars are 95% bootstrap intervals.",
  });
  legend(left, [attention, stateSpace]);
  mount(html("div", { class: "chart" }, left), (container) =>
    renderLine(container, { height: 300, x: xSpec, y: ySpec, series: [attention, stateSpace] })
  );

  const right = figure(grid, {
    title: "Paired family difference",
    caption:
      "The gap is indistinguishable from zero at the two smallest pairs and appears from 790M vs 1B onward. " +
      "The smallest Pythia has an oracle ceiling of 0.009, so its flat curve is a capability floor, not architecture invariance.",
  });
  legend(right, [difference]);
  mount(html("div", { class: "chart" }, right), (container) =>
    renderLine(container, {
      height: 300,
      x: xSpec,
      y: { ...ySpec, label: "Paired primacy difference (pp)" },
      series: [difference],
    })
  );

  table(
    parent,
    [
      { label: "Pair" },
      { label: "Attention primacy", numeric: true },
      { label: "State-space primacy", numeric: true },
      { label: "Difference", numeric: true },
      { label: "Holm p", numeric: true },
    ],
    DATA.scale.map((entry) => [
      entry.pair,
      effectCell(entry.attention.primacy),
      effectCell(entry.state_space.primacy),
      effectCell(entry.difference),
      pval(entry.difference.p),
    ]),
    "Phase 4 scale sweep. Percentage points, with 95% bootstrap intervals."
  );
}

function buildMechanisms() {
  const parent = $("#mechanisms");
  const grid = html("div", { class: "grid-2" }, parent);

  const sinkBox = figure(grid, {
    title: "Late-layer attention sink vs primacy",
    caption:
      "Token-0 attention share in the final layer, against the measured primacy effect, across five Pythia scales. " +
      "Correlational: Pythia 410M carries substantial mid-network sink mass with little final-layer sink and no significant primacy edge.",
  });
  mount(html("div", { class: "chart" }, sinkBox), (container) =>
    renderScatter(container, {
      height: 300,
      points: DATA.mechanisms.sink.map((entry) => ({
        x: entry.final_layer_sink_mass,
        y: entry.primacy.estimate,
        lo: entry.primacy.ci[0],
        hi: entry.primacy.ci[1],
        label: entry.label.replace("Pythia ", ""),
        color: MIXER.attention,
      })),
      x: { label: "Final-layer sink mass", format: (value) => value.toFixed(2) },
      y: { label: "Primacy effect (pp)" },
    })
  );

  const probeBox = figure(grid, {
    title: "Position probe: storage vs utilisation",
    caption:
      "A frozen linear probe reading edge-versus-middle gold position out of mid-depth hidden states, with a " +
      "shuffled-label control. Mamba stores position at least as well as Pythia while showing no primacy edge in accuracy.",
  });
  const probeSeries = [
    {
      label: "Observed labels",
      color: INK,
      connect: false,
      points: DATA.mechanisms.probe.map((entry, index) => ({ x: index, y: entry.accuracy })),
    },
    {
      label: "Shuffled labels",
      color: "var(--ink-mute)",
      hollow: true,
      connect: false,
      points: DATA.mechanisms.probe.map((entry, index) => ({ x: index, y: entry.shuffled_accuracy })),
    },
  ];
  legend(probeBox, probeSeries);
  mount(html("div", { class: "chart" }, probeBox), (container) =>
    renderLine(container, {
      height: 300,
      x: {
        kind: "band",
        values: DATA.mechanisms.probe.map((entry, index) => index),
        format: (index) => DATA.mechanisms.probe[index].label,
        tipLabel: (index) => `${DATA.mechanisms.probe[index].label}, layer ${DATA.mechanisms.probe[index].layer}`,
        label: "Probed model",
      },
      y: { label: "5-fold probe accuracy", format: (value) => value.toFixed(2), domain: [0.42, 0.7] },
      rule: { y: 0.5, label: "chance" },
      connectors: true,
      dodge: false,
      series: probeSeries,
    })
  );

  const variantBox = figure(parent, {
    title: "Prompt sensitivity",
    caption:
      "Primacy under four query placements and two instruction templates, on a 200-question subset. " +
      "The bookend prompt creates a primacy edge in Mamba that the Liu baseline does not, and the instructional " +
      "template flattens Pythia's. Direction is robust; magnitude is not.",
  });
  const variantModels = ["pythia-2.8b", "mamba-2.8b"];
  legend(
    variantBox,
    variantModels.map((key) => ({
      label: DATA.models[key].label,
      color: MIXER[DATA.models[key].mixer],
    }))
  );
  mount(html("div", { class: "chart" }, variantBox), (container) =>
    renderBars(container, {
      labelWidth: 190,
      groups: DATA.mechanisms.variants.map((entry) => ({
        label: entry.label,
        bars: variantModels.map((key) => ({
          label: `${entry.label} · ${DATA.models[key].label}`,
          value: entry.primacy[key].estimate,
          lo: entry.primacy[key].ci[0],
          hi: entry.primacy[key].ci[1],
          color: MIXER[DATA.models[key].mixer],
        })),
      })),
      xLabel: "Primacy effect (percentage points)",
    })
  );
}

function buildTransfer() {
  const parent = $("#transfer");
  const tasks = [
    { key: "qa", label: "Multi-document QA" },
    { key: "needle", label: "Needle, 2K tokens" },
  ];
  const groups = DATA.task_transfer.rows.map((row) => ({
    label: row.label,
    bars: tasks.map((task) => ({
      label: `${row.label} · ${task.label}`,
      value: row[task.key].primacy.estimate,
      lo: row[task.key].primacy.ci[0],
      hi: row[task.key].primacy.ci[1],
      note: pval(row[task.key].primacy.p),
      color: MIXER[DATA.models[row.model].mixer],
      hollow: task.key === "needle",
    })),
  }));
  const box = figure(parent, {
    title: "Primacy on multi-document QA and on synthetic needle retrieval",
    caption:
      "RULER <code>niah_single_1</code> at 2,048 tokens, 50 needle instances per model. Pythia reproduces its primacy " +
      "arm on the synthetic task. Both Mamba models answer every needle depth correctly, so their zero edges are " +
      "saturation rather than position invariance and the task cannot support a mixer comparison there.",
  });
  legend(box, [
    { label: "Multi-document QA", color: INK },
    { label: "Needle, 2K tokens", color: INK, faded: true },
  ]);
  mount(html("div", { class: "chart" }, box), (container) =>
    renderBars(container, {
      groups,
      barHeight: 17,
      labelWidth: 160,
      xLabel: "Primacy effect (percentage points)",
    })
  );

  table(
    parent,
    [
      { label: "Model" },
      { label: "QA primacy", numeric: true },
      { label: "Needle primacy", numeric: true },
      { label: "Needle accuracy", numeric: true },
    ],
    DATA.task_transfer.rows.map((row) => [
      row.label,
      effectCell(row.qa.primacy),
      effectCell(row.needle.primacy),
      acc(row.needle_accuracy),
    ]),
    "Phase 6 task transfer."
  );
}

function buildCalibration() {
  const parent = $("#calibration");
  const grid = html("div", { class: "grid-2" }, parent);

  const conditions = DATA.calibration.conditions;
  const conditionBox = figure(grid, {
    title: "QA calibration conditions",
    caption:
      "Closed book anchors guessing and memorisation; oracle anchors answerability with perfect retrieval. " +
      "Gold-first beats gold-middle by 11.5 points before any model comparison is attempted.",
  });
  mount(html("div", { class: "chart" }, conditionBox), (container) =>
    renderBars(container, {
      barHeight: 22,
      labelWidth: 110,
      groups: conditions.map((entry) => ({
        label: entry.label,
        bars: [
          {
            label: `Pythia 2.8B · ${entry.label}`,
            value: entry.accuracy,
            lo: entry.ci[0],
            hi: entry.ci[1],
            color: MIXER.attention,
          },
        ],
      })),
      xLabel: "Answer accuracy",
      tickFormat: (value) => value.toFixed(2),
      markFormat: acc,
      valueLabel: "Accuracy",
    })
  );

  const kvBox = figure(grid, {
    title: "Key-value positive control",
    caption:
      "A synthetic retrieval task with a known position effect. The harness recovers a 38-point edge-minus-middle " +
      "gap, which is what licenses reading a null result on the QA task as a null.",
  });
  mount(html("div", { class: "chart" }, kvBox), (container) =>
    renderLine(container, {
      height: 290,
      x: {
        kind: "linear",
        values: DATA.calibration.kv_curve.map((entry) => entry.slot),
        label: "Key-value slot",
        tipLabel: (value) => `Slot ${value}`,
      },
      y: { label: "Exact-match accuracy", format: (value) => value.toFixed(2), tipFormat: acc },
      series: [
        {
          label: "Pythia 2.8B",
          color: MIXER.attention,
          points: DATA.calibration.kv_curve.map((entry) => ({ x: entry.slot, y: entry.accuracy })),
        },
      ],
    })
  );
}

function buildPhases() {
  const parent = $("#phases");
  DATA.phases.forEach((phase, index) => {
    const node = html("div", { class: "phase" }, parent);
    const left = html("div", {}, node);
    html("div", { class: "idx", text: `Phase ${index + 1}` }, left);
    html("div", { class: "name", text: phase.name }, left);
    html("div", { class: "tag", text: phase.status }, left);
    const right = html("div", {}, node);
    html("div", { class: "q", text: phase.question }, right);
    html("div", { class: "a", text: phase.finding }, right);
  });
}

/* ---------- boot ---------- */

async function boot() {
  DATA = await (await fetch("data/results.json")).json();
  buildHeadline();
  buildPanels();
  buildContrasts();
  buildScale();
  buildMechanisms();
  buildTransfer();
  buildCalibration();
  buildPhases();
  flush();
}

boot();
