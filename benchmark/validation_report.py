# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:52:54 2026

@author: Admin
"""

import html
import math
import pathlib
import statistics
import tempfile
import webbrowser

_SVG_STYLE = """<style>
svg.pl-root{--pl-bg:#ffffff;--pl-fg:#1f2328;--pl-muted:#656d76;--pl-grid:#e1e4e8;--pl-axis:#8c959f;--pl-point:#1f77b4;--pl-accent:#d1242f;--pl-outlier:#f0883e}
@media (prefers-color-scheme:dark){svg.pl-root{--pl-bg:#0d1117;--pl-fg:#e6edf3;--pl-muted:#8b949e;--pl-grid:#262c36;--pl-axis:#6e7681;--pl-point:#58a6ff;--pl-accent:#ff7b72;--pl-outlier:#ffa657}}
.pl-bg{fill:var(--pl-bg)}
.pl-text{fill:var(--pl-fg);font:12px system-ui,sans-serif}
.pl-title{font-size:14px;font-weight:600}
.pl-small{font-size:10px}
.pl-grid{stroke:var(--pl-grid);stroke-width:1}
.pl-axis{stroke:var(--pl-axis);stroke-width:1;fill:none}
.pl-line{stroke:var(--pl-point);stroke-width:1.5;fill:none}
.pl-point{fill:var(--pl-point);fill-opacity:.7}
.pl-outlier{fill:var(--pl-outlier);stroke:var(--pl-fg);stroke-width:.5}
.pl-accent{stroke:var(--pl-accent);stroke-width:1.5;fill:none}
.pl-accent-fill{fill:var(--pl-accent)}
.pl-ref{stroke:var(--pl-muted);stroke-width:1;stroke-dasharray:5 4;fill:none}
</style>"""


def _f(v: float) -> str:
    return f"{v:.1f}"


def _padded(values: list, pad: float = 0.05) -> tuple:
    """(low, high) of the values, widened by pad on each side."""
    lo, hi = min(values), max(values)
    if hi == lo:
        lo, hi = lo - 1, hi + 1
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


def _nice_ticks(lo: float, hi: float, target: int = 6) -> list:
    """Round tick positions (1, 2 or 5 times a power of ten) inside [lo, hi]."""
    raw = (hi - lo) / target
    magnitude = 10 ** math.floor(math.log10(raw))
    step = magnitude
    for factor in (1, 2, 5, 10):
        step = factor * magnitude
        if step >= raw:
            break
    first = math.ceil(lo / step - 1e-9)
    last = math.floor(hi / step + 1e-9)
    return [round(i * step, 10) for i in range(first, last + 1)]


class _Plot:
    """Minimal SVG scatter/line plot: axes, grid, ticks, legend below the plot."""
    LEFT, RIGHT, TOP, BOTTOM, ROW = 70, 20, 40, 55, 18

    def __init__(self, width, height, xlim, ylim, xlabel, ylabel, title):
        self.w, self.h = width, height
        self.xlim, self.ylim = xlim, ylim
        self.xlabel, self.ylabel, self.title = xlabel, ylabel, title
        self.body, self.legend = [], []

    def px(self, x):
        x0, x1 = self.xlim
        return self.LEFT + (x - x0) / (x1 - x0) * (self.w - self.LEFT - self.RIGHT)

    def py(self, y):
        y0, y1 = self.ylim
        return self.h - self.BOTTOM - (y - y0) / (y1 - y0) * (self.h - self.TOP - self.BOTTOM)

    def line(self, x1, y1, x2, y2, cls):
        self.body.append(
            f'<line class="{cls}" x1="{_f(self.px(x1))}" y1="{_f(self.py(y1))}" '
            f'x2="{_f(self.px(x2))}" y2="{_f(self.py(y2))}"/>')

    def polyline(self, xs, ys, cls):
        pts = " ".join(f"{_f(self.px(x))},{_f(self.py(y))}" for x, y in zip(xs, ys))
        self.body.append(f'<polyline class="{cls}" points="{pts}"/>')

    def dot(self, x, y, cls, r=3, tip=""):
        title = f"<title>{html.escape(tip)}</title>" if tip else ""
        self.body.append(f'<circle class="{cls}" cx="{_f(self.px(x))}" '
                         f'cy="{_f(self.py(y))}" r="{r}">{title}</circle>')

    def text(self, x, y, label):
        """Small label beside a point; flips to the left near the right edge."""
        px = self.px(x)
        anchor, dx = ("end", -6) if px > self.w * 0.8 else ("start", 6)
        self.body.append(
            f'<text class="pl-text pl-small" x="{_f(px + dx)}" '
            f'y="{_f(self.py(y) - 5)}" text-anchor="{anchor}">{html.escape(label)}</text>')

    def add_legend(self, marker, cls, label):
        self.legend.append((marker, cls, label))

    def svg(self) -> str:
        total = self.h + (self.ROW * len(self.legend) + 10 if self.legend else 0)
        left, right = self.LEFT, self.w - self.RIGHT
        top, bottom = self.TOP, self.h - self.BOTTOM
        mid_y = (top + bottom) / 2

        o = [f'<svg xmlns="http://www.w3.org/2000/svg" class="pl-root" '
             f'viewBox="0 0 {self.w} {total}" width="{self.w}" height="{total}" '
             f'style="max-width:100%;height:auto" role="img">',
             _SVG_STYLE,
             f'<rect class="pl-bg" width="{self.w}" height="{total}"/>']

        for t in _nice_ticks(*self.xlim):
            x = _f(self.px(t))
            o.append(f'<line class="pl-grid" x1="{x}" y1="{top}" x2="{x}" y2="{bottom}"/>')
            o.append(f'<text class="pl-text pl-small" x="{x}" y="{bottom + 16}" '
                     f'text-anchor="middle">{t:g}</text>')
        for t in _nice_ticks(*self.ylim):
            y = _f(self.py(t))
            o.append(f'<line class="pl-grid" x1="{left}" y1="{y}" x2="{right}" y2="{y}"/>')
            o.append(f'<text class="pl-text pl-small" x="{left - 8}" y="{_f(self.py(t) + 3)}" '
                     f'text-anchor="end">{t:g}</text>')

        o += self.body
        o.append(f'<rect class="pl-axis" x="{left}" y="{top}" '
                 f'width="{right - left}" height="{bottom - top}"/>')
        if self.title:
            o.append(f'<text class="pl-text pl-title" x="{left}" y="24">'
                     f'{html.escape(self.title)}</text>')
        o.append(f'<text class="pl-text" x="{(left + right) / 2}" y="{bottom + 40}" '
                 f'text-anchor="middle">{html.escape(self.xlabel)}</text>')
        o.append(f'<text class="pl-text" transform="rotate(-90 18 {mid_y})" x="18" y="{mid_y}" '
                 f'text-anchor="middle">{html.escape(self.ylabel)}</text>')

        for i, (marker, cls, label) in enumerate(self.legend):
            yc = self.h + 6 + i * self.ROW + 9
            if marker == "line":
                o.append(f'<line class="{cls}" x1="{left}" y1="{yc}" x2="{left + 20}" y2="{yc}"/>')
            else:
                o.append(f'<circle class="{cls}" cx="{left + 10}" cy="{yc}" r="4"/>')
            o.append(f'<text class="pl-text" x="{left + 28}" y="{yc + 4}">{html.escape(label)}</text>')

        o.append("</svg>")
        return "\n".join(o)


def _output(svg: str, path: str = None) -> None:
    """Saves the svg to path, or opens it in the default browser if path is None."""
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False,
                                     encoding="utf-8") as f:
        f.write('<!DOCTYPE html><html><head><meta charset="utf-8"></head>'
                f'<body style="margin:0">{svg}</body></html>')
    webbrowser.open(pathlib.Path(f.name).as_uri())


def _tip(name, t, p, value, unit):
    return f"{name}: truth {t:g}, prediction {p:g}, {value:+.1f} {unit}"


# ----------------------------------------------------------------------
# sMAPE against threshold
# ----------------------------------------------------------------------

def svg_smape(results: dict, title: str = "sMAPE against score threshold") -> str:
    """Returns the sMAPE-against-threshold plot as an svg string."""
    thresholds = sorted(results)
    values = [results[t] for t in thresholds]
    best = thresholds[values.index(min(values))]
    best_value = results[best]

    plot = _Plot(800, 480, _padded(thresholds, 0.02), _padded(values),
                 "Score threshold", "sMAPE (%)", title)
    plot.line(best, plot.ylim[0], best, plot.ylim[1], "pl-ref")
    plot.polyline(thresholds, values, "pl-line")
    plot.dot(best, best_value, "pl-accent-fill", 5,
             f"threshold {best:g}, sMAPE {best_value:.2f}")
    plot.add_legend("dot", "pl-accent-fill",
                    f"Best: threshold = {best:g}, sMAPE = {best_value:.2f}")
    return plot.svg()


def render_smape(results: dict, path: str = None,
                 title: str = "sMAPE against score threshold") -> None:
    """
    Plots sMAPE for every threshold and marks the lowest value.

    results : {threshold: smape}, as returned by sweep_thresholds().
    path    : svg file to write. If None, the plot opens in the web browser.
    """
    _output(svg_smape(results, title), path)


# ----------------------------------------------------------------------
# Relative Bland-Altman
# ----------------------------------------------------------------------
def _ba_rows(pred: dict, truth: dict, min: int, max: int) -> list:
    """
    Builds the points of a relative Bland-Altman plot as a list of
    (sample_name, truth, prediction, mean, relative_difference_in_per_cent).

    pred, truth : dicts of sample name -> count. Only names present in both are used.
    min, max    : inclusive integer limits applied to the ground-truth counts.
    """
    rows = []
    for name in truth.keys() & pred.keys():
        t, p = truth[name], pred[name]
        if not (min <= t <= max):
            continue
        if t == 0 and p == 0:                  # relative difference undefined
            continue
        mean = (p + t) / 2
        rows.append((name, t, p, mean, 100 * (p - t) / mean))

    if len(rows) < 2:
        raise ValueError("At least two observations are required.")
    return rows


def _split_outliers(rows: list, bias: float, n: int) -> tuple:
    """Returns the n rows furthest above and the n furthest below the bias."""
    ordered = sorted(rows, key=lambda row: row[4])
    above = [row for row in reversed(ordered) if row[4] > bias][:n]
    below = [row for row in ordered if row[4] < bias][:n]
    return above, below


def svg_ba(pred: dict, truth: dict, min: int, max: int,
           title: str = None, n_outliers: int = 0) -> tuple:
    """
    Returns (svg, bias, lower_limit, upper_limit, above, below).
    See render_ba() for the meaning of the arguments and the lists.
    """
    rows = _ba_rows(pred, truth, min, max)
    diffs = [row[4] for row in rows]

    bias = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    upper, lower = bias + 1.96 * sd, bias - 1.96 * sd

    plot = _Plot(800, 480,
                 _padded([row[3] for row in rows]),         # xlim
                 _padded(diffs + [lower, upper]),           # ylim
                 "Mean of prediction and ground truth (colonies)",   # xlabel
                 "Relative difference (%)",                          # ylabel
                 "Relative Bland-Altman plot" if title is None else title)
    x0, x1 = plot.xlim
    plot.line(x0, upper, x1, upper, "pl-ref")
    plot.line(x0, lower, x1, lower, "pl-ref")
    plot.line(x0, bias, x1, bias, "pl-accent")
    for name, t, p, mean, diff in rows:
        plot.dot(mean, diff, "pl-point", 3, _tip(name, t, p, diff, "%"))

    above, below = [], []
    if n_outliers > 0:
        above_rows, below_rows = _split_outliers(rows, bias, n_outliers)
        for name, t, p, mean, diff in above_rows + below_rows:
            plot.dot(mean, diff, "pl-outlier", 4.5, _tip(name, t, p, diff, "%"))
            plot.text(mean, diff, name)
        above = [(r[0], r[1], r[2], r[4]) for r in above_rows]
        below = [(r[0], r[1], r[2], r[4]) for r in below_rows]
        plot.add_legend("dot", "pl-outlier", f"Outliers ({n_outliers} per side)")

    plot.add_legend("line", "pl-accent", f"Bias: {bias:.2f} %")
    plot.add_legend("line", "pl-ref", f"Upper limit: {upper:.2f} %")
    plot.add_legend("line", "pl-ref", f"Lower limit: {lower:.2f} %")
    return plot.svg(), bias, lower, upper, above, below


def render_ba(pred: dict, truth: dict, min: int, max: int,
              path: str = None, title: str = None,
              n_outliers: int = 0) -> tuple:
    """
    Plots a relative Bland-Altman plot for two sets of counts.

    x-axis : mean of prediction and ground truth, (pred + truth) / 2.
    y-axis : difference relative to that mean, 100 * (pred - truth) / mean (%).
    Lines  : bias (mean difference) and limits of agreement (bias +/- 1.96 SD).

    pred, truth : dicts of sample name -> count.
    min, max    : inclusive integer limits applied to the ground-truth counts.
    path        : svg file to write. If None, the plot opens in the web browser.
    n_outliers  : outliers highlighted on each side of the bias (0 = none).

    Returns (bias, lower_limit, upper_limit, above, below).
    above, below : lists of (sample_name, truth, prediction,
                   relative_difference_in_per_cent), largest deviation first.
    """
    svg, bias, lower, upper, above, below = svg_ba(
        pred, truth, min, max, title, n_outliers)
    _output(svg, path)
    return bias, lower, upper, above, below


# ----------------------------------------------------------------------
# Deming regression
# ----------------------------------------------------------------------

def _deming_fit(xs: list, ys: list, delta: float) -> tuple:
    """Returns (slope, intercept) of the Deming regression."""
    count = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs) / (count - 1)
    syy = sum((y - my) ** 2 for y in ys) / (count - 1)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (count - 1)
    if sxy == 0:
        raise ValueError("Covariance is zero; the Deming slope is undefined.")
    d = syy - delta * sxx
    slope = (d + math.sqrt(d ** 2 + 4 * delta * sxy ** 2)) / (2 * sxy)
    return slope, my - slope * mx


def svg_deming(pred: dict, truth: dict, min: int, max: int,
               title: str = None, n_outliers: int = 0,
               delta: float = 1.0) -> tuple:
    """
    Returns (svg, slope, intercept, above, below).
    See render_deming() for the meaning of the arguments and the lists.
    """
    if delta <= 0:
        raise ValueError("'delta' must be positive.")

    rows = _ba_rows(pred, truth, min, max)
    xs = [row[1] for row in rows]                 # ground truth
    ys = [row[2] for row in rows]                 # prediction
    slope, intercept = _deming_fit(xs, ys, delta)

    xa, xb = sorted(xs)[0], sorted(xs)[-1]
    ya, yb = intercept + slope * xa, intercept + slope * xb
    lim = _padded(xs + ys + [ya, yb])             # same limits on both axes

    plot = _Plot(600, 600, lim, lim, "Ground truth (colonies)",
             "Prediction (colonies)",
             "Deming regression" if title is None else title)
    plot.line(lim[0], lim[0], lim[1], lim[1], "pl-ref")
    plot.line(xa, ya, xb, yb, "pl-accent")
    for name, t, p, _, _ in rows:
        plot.dot(t, p, "pl-point", 3,
                 _tip(name, t, p, p - (intercept + slope * t), "colonies from the line"))

    above, below = [], []
    if n_outliers > 0:
        fit_rows = [(name, t, p, t, p - (intercept + slope * t))
                    for name, t, p, _, _ in rows]
        above_rows, below_rows = _split_outliers(fit_rows, 0.0, n_outliers)
        for name, t, p, _, resid in above_rows + below_rows:
            plot.dot(t, p, "pl-outlier", 4.5,
                     _tip(name, t, p, resid, "colonies from the line"))
            plot.text(t, p, name)
        above = [(r[0], r[1], r[2], r[4]) for r in above_rows]
        below = [(r[0], r[1], r[2], r[4]) for r in below_rows]
        plot.add_legend("dot", "pl-outlier", f"Outliers ({n_outliers} per side)")

    plot.add_legend("line", "pl-accent", f"Deming: y = {slope:.3f}x {intercept:+.2f}")
    plot.add_legend("line", "pl-ref", "Identity (y = x)")
    return plot.svg(), slope, intercept, above, below


def render_deming(pred: dict, truth: dict, min: int, max: int,
                  path: str = None, title: str = None,
                  n_outliers: int = 0, delta: float = 1.0) -> tuple:
    """
    Plots ground truth against prediction with a Deming regression line.

    pred, truth : dicts of sample name -> count.
    min, max    : inclusive integer limits applied to the ground-truth counts.
    path        : svg file to write. If None, the plot opens in the web browser.
    n_outliers  : outliers highlighted on each side of the line (0 = none).
    delta       : ratio of the error variance of the prediction to that of the
                  ground truth. 1.0 gives orthogonal regression.

    Returns (slope, intercept, above, below).
    above, below : lists of (sample_name, truth, prediction, residual),
                   largest deviation first; residual in colonies.
    """
    svg, slope, intercept, above, below = svg_deming(
        pred, truth, min, max, title, n_outliers, delta)
    _output(svg, path)
    return slope, intercept, above, below

import datetime
import os
from string import Template


def _outlier_table(items: list, value_header: str, fmt: str) -> str:
    """HTML table of (sample_name, truth, prediction, value) tuples."""
    if not items:
        return '<p class="muted">None.</p>'
    rows = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{t:g}</td><td>{p:g}</td>"
        f"<td>{v:{fmt}}</td></tr>"
        for name, t, p, v in items)
    return (f"<table><tr><th>Sample</th><th>Truth</th><th>Prediction</th>"
            f"<th>{value_header}</th></tr>{rows}</table>")


def build_report(ev: dict, out_path: str, min: int, max: int,
                 n_outliers: int = 5, delta: float = 1.0,
                 template_path: str = None,
                 title: str = "Colony counting validation") -> str:
    """
    Renders a self-contained HTML report from the dictionary returned by
    evaluate_thresholds(). Supports both full optimization results and
    minimal dictionaries containing only best_smape, pred, and truth.
    """
    if template_path is None:
        folder = os.path.dirname(os.path.abspath(globals().get("__file__", "")))
        template_path = os.path.join(folder, "report_template.html")
    
    with open(template_path, encoding="utf-8") as f:
        template_content = f.read()

    pred, truth = ev["pred"], ev["truth"]
    
    # Check if full threshold optimization info is available
    has_threshold_info = "best_threshold" in ev and "results" in ev

    if has_threshold_info:
        best = ev["best_threshold"]
        smape_svg = svg_smape(ev["results"], title="")
        best_threshold_str = f"{best:g}"
        best_smape_str = f"{ev['best_smape']:.2f}"
    else:
        # Fallback when only best_smape, pred, and truth are provided:
        # Show just the sMAPE value prominently instead of the plot and threshold stats.
        best_threshold_str = "N/A"
        best_smape_str = f"{ev['best_smape']:.2f}"
        smape_svg = f"""
        <div style="padding: 40px; text-align: center; background: rgba(0,0,0,0.02); border-radius: 8px; border: 1px solid var(--pl-grid, #e1e4e8);">
            <div style="font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--pl-muted, #656d76); margin-bottom: 8px;">Evaluation sMAPE</div>
            <div style="font-size: 42px; font-weight: 700; color: var(--pl-fg, #1f2328);">{ev['best_smape']:.2f}%</div>
        </div>
        """

    ba_svg, bias, lower, upper, ba_above, ba_below = svg_ba(
        pred, truth, min, max, "", n_outliers)
    deming_svg, slope, intercept, dm_above, dm_below = svg_deming(
        pred, truth, min, max, "", n_outliers, delta)

    values = {
        "title": html.escape(title),
        "generated": datetime.datetime.now().strftime("%d %B %Y, %H:%M"),
        "best_threshold": best_threshold_str,
        "best_smape": best_smape_str,
        "matched": len(truth.keys() & pred.keys()),
        "used": len(_ba_rows(pred, truth, min, max)),
        "min": min,
        "max": max,
        "ba_bias": f"{bias:+.2f}",
        "ba_lower": f"{lower:.2f}",
        "ba_upper": f"{upper:.2f}",
        "ba_above_table": _outlier_table(ba_above, "Difference (%)", "+.1f"),
        "ba_below_table": _outlier_table(ba_below, "Difference (%)", "+.1f"),
        "dm_slope": f"{slope:.3f}",
        "dm_intercept": f"{intercept:+.2f}",
        "delta": f"{delta:g}",
        "dm_above_table": _outlier_table(dm_above, "Residual (colonies)", "+.1f"),
        "dm_below_table": _outlier_table(dm_below, "Residual (colonies)", "+.1f"),
        "smape_svg": smape_svg,
        "ba_svg": ba_svg,
        "deming_svg": deming_svg,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(Template(template_content).safe_substitute(values))

    print(f"Report written: {out_path}")
    return out_path


# %% STATISTIC TBD


def _t_critical(df: int, confidence: float = 0.95) -> float:
    """
    Two-sided critical value of Student's t distribution (standard library only).
    Cornish-Fisher expansion of the normal quantile; accurate to about three
    decimals from df = 10 upwards.
    """
    z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2)
    return (z
            + (z**3 + z) / (4 * df)
            + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3))


def _trend(xs: list, ys: list) -> dict:
    """Linear regression of ys on xs, with a t-test of the slope at the 5 % level."""
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if n < 3 or sxx == 0 or syy == 0:
        return {"slope": None, "r": None, "t": None, "significant": None}
    r = sxy / math.sqrt(sxx * syy)
    t = r * math.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else math.inf
    return {"slope": sxy / sxx, "r": r, "t": t,
            "significant": abs(t) > _t_critical(n - 2)}


def ba_statistics(prediction: dict, truth: dict, threshold: float,
                  min: int, max: int) -> dict:
    """
    Statistics for the relative Bland-Altman plot at one score threshold.
    The samples are the same as in render_ba().

    Returns a dictionary with:
      n                : number of samples
      bias             : mean relative difference (%)
      bias_ci          : 95 % confidence interval of the bias
      bias_significant : True if the bias interval excludes 0
      sd               : standard deviation of the relative differences
      lower, upper     : limits of agreement (bias +/- 1.96 SD)
      lower_ci, upper_ci : 95 % confidence intervals of the two limits
      n_outside        : samples outside the limits of agreement
      pct_outside      : the same in per cent (about 5 % is expected)
      skewness, excess_kurtosis : both 0 for a normal distribution
      proportional     : trend of the relative difference against the mean
      spread           : trend of |difference - bias| against the mean
                         (each: slope, r, t, significant)
    """
    rows = _ba_rows(prediction, truth, threshold, min, max)
    means = [row[3] for row in rows]
    diffs = [row[4] for row in rows]

    n = len(diffs)
    bias = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    t_crit = _t_critical(n - 1)

    se_bias = sd / math.sqrt(n)
    se_limit = sd * math.sqrt(1 / n + 1.96**2 / (2 * (n - 1)))
    upper, lower = bias + 1.96 * sd, bias - 1.96 * sd
    n_outside = sum(1 for d in diffs if d < lower or d > upper)

    m2 = sum((d - bias) ** 2 for d in diffs) / n
    if m2 > 0:
        skewness = sum((d - bias) ** 3 for d in diffs) / n / m2 ** 1.5
        kurtosis = sum((d - bias) ** 4 for d in diffs) / n / m2 ** 2 - 3
    else:
        skewness = kurtosis = None

    bias_ci = (bias - t_crit * se_bias, bias + t_crit * se_bias)

    return {
        "n": n,
        "bias": bias,
        "bias_ci": bias_ci,
        "bias_significant": not (bias_ci[0] <= 0 <= bias_ci[1]),
        "sd": sd,
        "lower": lower,
        "upper": upper,
        "lower_ci": (lower - t_crit * se_limit, lower + t_crit * se_limit),
        "upper_ci": (upper - t_crit * se_limit, upper + t_crit * se_limit),
        "n_outside": n_outside,
        "pct_outside": 100 * n_outside / n,
        "skewness": skewness,
        "excess_kurtosis": kurtosis,
        "proportional": _trend(means, diffs),
        "spread": _trend(means, [abs(d - bias) for d in diffs]),
    }


# Usage
"""
s = ba_statistics(prediction, truth, best, min=11, max=300)

print(f"n = {s['n']}")
print(f"Bias: {s['bias']:+.2f} % (95 % CI {s['bias_ci'][0]:.2f} to {s['bias_ci'][1]:.2f})")
print(f"Upper limit: {s['upper']:.2f} % (95 % CI {s['upper_ci'][0]:.2f} to {s['upper_ci'][1]:.2f})")
print(f"Lower limit: {s['lower']:.2f} % (95 % CI {s['lower_ci'][0]:.2f} to {s['lower_ci'][1]:.2f})")
print(f"Outside the limits: {s['n_outside']} ({s['pct_outside']:.1f} %)")
print(f"Skewness: {s['skewness']:.2f}, excess kurtosis: {s['excess_kurtosis']:.2f}")
print(f"Trend of difference against mean: slope = {s['proportional']['slope']:.4f}, "
      f"significant = {s['proportional']['significant']}")
print(f"Trend of spread against mean: slope = {s['spread']['slope']:.4f}, "
      f"significant = {s['spread']['significant']}")
"""

# %%


def _jackknife_se(loo: list) -> float:
    """Jackknife standard error from the leave-one-out estimates."""
    n = len(loo)
    m = sum(loo) / n
    return math.sqrt((n - 1) / n * sum((v - m) ** 2 for v in loo))


def deming_statistics(prediction: dict, truth: dict, threshold: float,
                      min: int, max: int, delta: float = 1.0,
                      levels: list = None,
                      deltas: tuple = (0.5, 1.0, 2.0)) -> dict:
    """
    Statistics for the Deming regression at one score threshold.
    The samples are the same as in render_deming() and render_ba().

    delta  : assumed ratio of the error variance of the prediction to that
             of the ground truth.
    levels : ground-truth counts at which the bias (prediction - truth) is
             estimated, for example [20, 50, 100, 200].
    deltas : values of delta used for the sensitivity analysis.

    Returns a dictionary with:
      n                  : number of samples
      slope, intercept   : Deming estimates
      slope_ci, intercept_ci : 95 % confidence intervals (jackknife)
      slope_differs_from_1, intercept_differs_from_0 : True if the interval
                           excludes the ideal value
      bias_at            : {level: (bias, (lower, upper))}, if levels is given
      r                  : Pearson correlation of prediction and ground truth
      residual_sd        : standard deviation of the vertical residuals (colonies)
      n_outside, pct_outside : samples with |residual| above 1.96 SD
      skewness, excess_kurtosis : of the residuals (both 0 for a normal distribution)
      spread             : trend of |residual| against the ground truth
                           (slope, r, t, significant)
      influential        : the three samples whose removal changes the slope
                           most, as (sample_name, slope_without_it)
      delta_sensitivity  : {delta: slope}
    """
    if delta <= 0:
        raise ValueError("'delta' must be positive.")

    rows = _ba_rows(prediction, truth, threshold, min, max)
    names = [row[0] for row in rows]
    xs = [row[1] for row in rows]                 # ground truth
    ys = [row[2] for row in rows]                 # prediction
    n = len(rows)
    if n < 3:
        raise ValueError("At least three observations are required.")

    fit = _deming_fit(xs, ys, delta)
    if fit is None:
        raise ValueError("Covariance is zero; the Deming slope is undefined.")
    slope, intercept = fit

    # Leave-one-out fits, shared by the intervals and the influence check
    loo = []
    for i in range(n):
        fit_i = _deming_fit(xs[:i] + xs[i + 1:], ys[:i] + ys[i + 1:], delta)
        if fit_i is None:
            raise ValueError("Covariance is zero after leaving out a sample.")
        loo.append(fit_i)

    t_crit = _t_critical(n - 2)
    slope_se = _jackknife_se([f[0] for f in loo])
    intercept_se = _jackknife_se([f[1] for f in loo])
    slope_ci = (slope - t_crit * slope_se, slope + t_crit * slope_se)
    intercept_ci = (intercept - t_crit * intercept_se,
                    intercept + t_crit * intercept_se)

    # Bias at chosen counts
    bias_at = {}
    for level in levels or []:
        estimate = intercept + (slope - 1) * level
        se = _jackknife_se([a + (b - 1) * level for b, a in loo])
        bias_at[level] = (estimate, (estimate - t_crit * se,
                                     estimate + t_crit * se))

    # Residuals
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    residual_sd = math.sqrt(sum(e ** 2 for e in residuals) / (n - 2))
    n_outside = sum(1 for e in residuals if abs(e) > 1.96 * residual_sd)

    m2 = sum(e ** 2 for e in residuals) / n
    if m2 > 0:
        skewness = sum(e ** 3 for e in residuals) / n / m2 ** 1.5
        kurtosis = sum(e ** 4 for e in residuals) / n / m2 ** 2 - 3
    else:
        skewness = kurtosis = None

    # Pearson correlation
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r = sxy / math.sqrt(sxx * syy)

    # Influence: change of the slope when one sample is left out
    ranked = sorted(range(n), key=lambda i: abs(loo[i][0] - slope), reverse=True)
    influential = [(names[i], loo[i][0]) for i in ranked[:3]]

    # Sensitivity to the assumed variance ratio
    sensitivity = {}
    for d in deltas:
        fit_d = _deming_fit(xs, ys, d)
        sensitivity[d] = fit_d[0] if fit_d else None

    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "slope_ci": slope_ci,
        "intercept_ci": intercept_ci,
        "slope_differs_from_1": not (slope_ci[0] <= 1 <= slope_ci[1]),
        "intercept_differs_from_0": not (intercept_ci[0] <= 0 <= intercept_ci[1]),
        "bias_at": bias_at,
        "r": r,
        "residual_sd": residual_sd,
        "n_outside": n_outside,
        "pct_outside": 100 * n_outside / n,
        "skewness": skewness,
        "excess_kurtosis": kurtosis,
        "spread": _trend(xs, [abs(e) for e in residuals]),
        "influential": influential,
        "delta_sensitivity": sensitivity,
    }


# Usage
"""
s = deming_statistics(prediction, truth, best, min=11, max=300,
                      levels=[20, 50, 100, 200])

print(f"n = {s['n']}")
print(f"Slope: {s['slope']:.3f} (95 % CI {s['slope_ci'][0]:.3f} to {s['slope_ci'][1]:.3f}), "
      f"differs from 1: {s['slope_differs_from_1']}")
print(f"Intercept: {s['intercept']:+.2f} (95 % CI {s['intercept_ci'][0]:.2f} to {s['intercept_ci'][1]:.2f}), "
      f"differs from 0: {s['intercept_differs_from_0']}")
for level, (bias, (lo, hi)) in s["bias_at"].items():
    print(f"Bias at {level} colonies: {bias:+.2f} (95 % CI {lo:.2f} to {hi:.2f})")
print(f"Pearson r: {s['r']:.3f}")
print(f"Residual SD: {s['residual_sd']:.2f} colonies; "
      f"outside 1.96 SD: {s['n_outside']} ({s['pct_outside']:.1f} %)")
print(f"Skewness: {s['skewness']:.2f}, excess kurtosis: {s['excess_kurtosis']:.2f}")
print(f"Trend of spread against ground truth: slope = {s['spread']['slope']:.4f}, "
      f"significant = {s['spread']['significant']}")
print("Most influential samples:", s["influential"])
print("Slope for other delta values:", s["delta_sensitivity"])
"""
