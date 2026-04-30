# -*- coding: utf-8 -*-

from datetime import datetime
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Global import
import argparse
import os
import gc
import glob
import sys
import time
import cv2

# Local import
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import main.tiling as tiling
from main.detect_frcnn_onnx import load_model, detect_on_tiles
from main.process import process_plate
from main.down import summarize_colonies


# ----------------------------------------------------------------------
# Logging — tees stdout to output_folder/pipeline_<timestamp>.log
# ----------------------------------------------------------------------

class _Tee:
    """Writes to both the original stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self._stdout = sys.stdout
        self._file = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = self

    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()

    def __getattr__(self, name):
        return getattr(self._stdout, name)


def setup_logging(output_folder):
    log_name = f"agar3000_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_folder, log_name)
    return _Tee(log_path), log_path


# ----------------------------------------------------------------------
# Timing helpers
# ----------------------------------------------------------------------

class StageTimer:
    """Tracks elapsed time for named pipeline stages."""

    def __init__(self):
        self.stages: list[dict] = []          # ordered list of completed stages
        self._start: float | None = None
        self._current_stage: str | None = None
        self.pipeline_start: float = time.perf_counter()

    def start(self, name: str):
        """Begin timing a new stage (automatically closes the previous one)."""
        if self._current_stage is not None:
            self._close_current()
        self._current_stage = name
        self._start = time.perf_counter()
        print(f"[TIMER] ▶ {name}")

    def stop(self):
        """Explicitly close the current stage."""
        if self._current_stage is not None:
            self._close_current()

    def _close_current(self):
        elapsed = time.perf_counter() - self._start
        self.stages.append({
            "name":    self._current_stage,
            "elapsed": elapsed,
        })
        print(f"[TIMER] ■ {self._current_stage}: {_fmt(elapsed)}")
        self._current_stage = None
        self._start = None

    def total(self) -> float:
        return time.perf_counter() - self.pipeline_start

    def aggregate(self) -> list[dict]:
        """Return per-stage-type totals, with per-image stages summed across all plates.

        Stage names that follow the pattern '[filename] Stage name' are stripped
        of the filename prefix and accumulated together.  Top-level stages (no
        bracket prefix) are kept as-is.  Order follows first appearance.
        """
        import re
        totals: dict[str, float] = {}
        order:  list[str]        = []
        pat = re.compile(r"^\[.+?\]\s+(.+)$")
        for s in self.stages:
            m = pat.match(s["name"])
            key = m.group(1) if m else s["name"]
            if key not in totals:
                order.append(key)
                totals[key] = 0.0
            totals[key] += s["elapsed"]
        return [{"name": k, "elapsed": totals[k]} for k in order]


def _fmt(seconds: float) -> str:
    """Human-readable duration: e.g. '1 m 03.4 s' or '8.23 s'."""
    if seconds >= 60:
        m = int(seconds // 60)
        s = seconds - m * 60
        return f"{m} m {s:05.2f} s"
    return f"{seconds:.3f} s"


# ----------------------------------------------------------------------
# HTML Report
# ----------------------------------------------------------------------

def _bar_width(val: float, max_val: float) -> float:
    return round(max(2.0, val / max_val * 100), 2) if max_val > 0 else 2.0


# ----------------------------------------------------------------------
# CSV export
# ----------------------------------------------------------------------

def save_timing_csv(timer: StageTimer, output_folder: str) -> str:
    """Write aggregated stage totals (summed across all plates) to timing.csv."""
    import csv

    csv_path = os.path.join(output_folder, "timing.csv")
    agg = timer.aggregate()
    total_s = timer.total()

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "elapsed_s", "pct_total"])
        for a in agg:
            pct = a["elapsed"] / total_s * 100 if total_s else 0.0
            w.writerow([a["name"], f"{a['elapsed']:.6f}", f"{pct:.4f}"])
        w.writerow(["TOTAL", f"{total_s:.6f}", "100.0000"])

    print(f"[CSV]    Timing CSV saved    → {csv_path}")
    return csv_path


def generate_timing_report(timer: StageTimer, output_folder: str,
                            config: dict, log_path: str):
    """Write a self-contained HTML timing report to output_folder."""

    total_s = timer.total()
    stages   = timer.stages
    agg      = timer.aggregate()
    max_agg  = max((a["elapsed"] for a in agg), default=1.0)

    # ---- aggregated rows ----
    agg_rows_html = ""
    for i, a in enumerate(agg):
        pct   = a["elapsed"] / total_s * 100 if total_s > 0 else 0
        bw    = _bar_width(a["elapsed"], max_agg)
        shade = "#1a1a2e" if i % 2 == 0 else "#16213e"
        agg_rows_html += f"""
        <tr style="background:{shade}">
          <td class="stage-name">{a['name']}</td>
          <td class="dur">{_fmt(a['elapsed'])}</td>
          <td class="pct-cell">{pct:.1f}%</td>
          <td class="bar-cell">
            <div class="bar agg-bar" style="width:{bw}%"></div>
          </td>
        </tr>"""

    # ---- config table ----
    cfg_rows = ""
    for k, v in config.items():
        cfg_rows += f"<tr><td class='cfg-key'>{k}</td><td class='cfg-val'>{v}</td></tr>"

    # ---- stage count summary ----
    n_images = config.get("Images found", "—")

    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agar3000 — Pipeline Timing Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  :root {{
    --bg:       #0d0d1a;
    --panel:    #1a1a2e;
    --accent:   #00e5ff;
    --accent2:  #7b2fff;
    --text:     #e0e0f0;
    --muted:    #7a7a9a;
    --border:   #2a2a4a;
    --success:  #00e676;
    --warn:     #ffea00;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'IBM Plex Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 2rem 1.5rem 4rem;
  }}

  /* ---- header ---- */
  header {{
    display: flex;
    align-items: flex-end;
    gap: 1.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.25rem;
    margin-bottom: 2.5rem;
  }}
  .logo-block {{ line-height: 1; }}
  .logo {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: -1px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .subtitle {{
    font-size: 0.78rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }}
  .meta {{
    margin-left: auto;
    text-align: right;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    line-height: 1.7;
  }}

  /* ---- KPI row ---- */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin-bottom: 2.5rem;
  }}
  .kpi {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.1rem 1.25rem;
    position: relative;
    overflow: hidden;
  }}
  .kpi::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }}
  .kpi-label {{ font-size: 0.7rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }}
  .kpi-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: var(--accent); }}
  .kpi-sub   {{ font-size: 0.7rem; color: var(--muted); margin-top: 2px; }}

  /* ---- section titles ---- */
  h2 {{
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
  }}

  /* ---- timing table ---- */
  .timing-wrap {{ margin-bottom: 2.5rem; overflow-x: auto; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }}
  thead th {{
    text-align: left;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
  }}
  td {{ padding: 0.55rem 0.75rem; vertical-align: middle; }}
  .stage-name {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: var(--text); white-space: nowrap; }}
  .dur        {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: var(--accent); white-space: nowrap; }}
  .pct-cell   {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--muted); white-space: nowrap; width: 3.5rem; }}
  .bar-cell   {{ width: 100%; min-width: 120px; }}
  .bar {{
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    min-width: 4px;
    transition: width 0.4s ease;
  }}
  .agg-bar {{
    background: linear-gradient(90deg, var(--success), #00b8d4);
  }}

  /* ---- section divider ---- */
  .section-divider {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2.5rem 0 1rem;
  }}
  .section-divider h2 {{ margin: 0; }}
  .section-divider::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }}

  /* ---- config table ---- */
  .config-wrap {{ margin-bottom: 2rem; overflow-x: auto; }}
  .cfg-table {{ width: auto; min-width: 400px; }}
  .cfg-key {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    white-space: nowrap;
    padding-right: 2rem;
  }}
  .cfg-val {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text);
  }}

  footer {{
    margin-top: 3rem;
    text-align: center;
    font-size: 0.7rem;
    color: var(--border);
    font-family: 'IBM Plex Mono', monospace;
  }}
</style>
</head>
<body>

<header>
  <div class="logo-block">
    <div class="logo">AGAR3000</div>
    <div class="subtitle">Pipeline Timing Report · v0.2</div>
  </div>
  <div class="meta">
    Generated: {report_time}<br>
    Log: {os.path.basename(log_path)}
  </div>
</header>

<!-- KPI row -->
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Total runtime</div>
    <div class="kpi-value">{_fmt(total_s)}</div>
    <div class="kpi-sub">wall-clock time</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Stage types</div>
    <div class="kpi-value">{len(agg)}</div>
    <div class="kpi-sub">distinct stages</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Images processed</div>
    <div class="kpi-value">{n_images}</div>
    <div class="kpi-sub">input files</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Slowest stage (agg.)</div>
    <div class="kpi-value" style="font-size:1rem">{max(agg, key=lambda a: a['elapsed'])['name'] if agg else '—'}</div>
    <div class="kpi-sub">{_fmt(max_agg)} cumulative</div>
  </div>
</div>

<!-- Aggregated summary -->
<div class="timing-wrap">
  <div class="section-divider"><h2>Stage timing — cumulative across all plates</h2></div>
  <table>
    <thead>
      <tr>
        <th>Stage type</th>
        <th>Cumulative</th>
        <th>% total</th>
        <th>Relative</th>
      </tr>
    </thead>
    <tbody>
      {agg_rows_html}
      <tr style="background:#0d0d1a; border-top:1px solid var(--border)">
        <td class="stage-name" style="color:var(--muted)">TOTAL</td>
        <td class="dur" style="color:var(--success)">{_fmt(total_s)}</td>
        <td class="pct-cell">100%</td>
        <td></td>
      </tr>
    </tbody>
  </table>
</div>

<!-- Configuration -->
<div class="config-wrap">
  <h2>Run configuration</h2>
  <table class="cfg-table">
    <tbody>
      {cfg_rows}
    </tbody>
  </table>
</div>

<footer>agar3000 · automated colony counter · timing report auto-generated</footer>

</body>
</html>"""

    report_path = os.path.join(output_folder, "timing_report.html")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[REPORT] Timing report saved → {report_path}")
    return report_path


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

def run_folder_pipeline(input_folder, output_folder, model="model/frcnn_norm.pt",
                        grid=(2, 2), overlap=0.2,
                        tol=5, scale=1024, score=0.25, extra=False,
                        mem_debug=False, no_crop=False, margin=1,
                        score_regression=None, timer: StageTimer = None):

    os.makedirs(output_folder, exist_ok=True)

    try:
        EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

        # ---- model loading ----
        timer.start("Model loading")
        rcnn = load_model(model)
        timer.stop()

        if os.path.isfile(input_folder):
            if os.path.splitext(input_folder)[1].lower() not in EXTS:
                raise ValueError(f"Unsupported file type: {input_folder}")
            img_paths = [input_folder]
        else:
            img_paths = []
            for e in (f"*{ext}" for ext in EXTS):
                img_paths.extend(glob.glob(os.path.join(input_folder, e)))

        print(f"Found images: {len(img_paths)}")

        for img_path in sorted(img_paths):
            img_name = os.path.basename(img_path)
            print("-----------------------------------------------------")
            print("-----------------------------------------------------")
            print(f"PROCESSING: {img_name} ({timestamp()})")
            print("-----------------------------------------------------")

            # ---- plate pre-processing ----
            plate = process_plate(img_path, margin=margin, timer=timer, img_name=img_name)
            if no_crop:
                plate.cropped = plate.image
            elif extra:
                ext_out = os.path.join(output_folder, "crop")
                os.makedirs(ext_out, exist_ok=True)
                cv2.imwrite(os.path.join(ext_out, f"{plate.sample_id}.png"), plate.cropped)
            timer.stop()

            # ---- tiling ----
            timer.start(f"[{img_name}] Tiling")
            tiles = tiling.make_tiles(plate.cropped, grid=grid, overlap=overlap)
            timer.stop()

            # ---- detection ----
            timer.start(f"[{img_name}] Colony detection")
            print(f"Detection started:  {plate.sample_id} ({timestamp()})")
            tiles = detect_on_tiles(tiles, rcnn, size=scale, conf=score, normalize=True, rgb=False)
            print(f"Detection finished: {plate.sample_id} ({timestamp()})")
            print("-----------------------------------------------------")
            timer.stop()

            if extra:
                ext_out = os.path.join(output_folder, "dup")
                tiling.show_all_tiles_with_boxes(tiles, key="tile_with_boxes",
                                                 cols=None,
                                                 output_folder=ext_out,
                                                 name=f"{plate.sample_id}_tiles")
                tiling.save_plate_tiles_to_csv(ext_out, tiles, name=plate.sample_id)

            # ---- deduplication ----
            timer.start(f"[{img_name}] Deduplication")
            tiling.resolve_duplicates_across_tiles(tiles, tol=tol, detections_key="colonies")
            timer.stop()
            print(f"Deduplication finished: {plate.sample_id} ({timestamp()})")

            if extra:
                ext_out = os.path.join(output_folder, "dedup")
                tiling.show_all_tiles_with_boxes(tiles, key="tile_with_boxes",
                                                 cols=None,
                                                 output_folder=ext_out,
                                                 name=f"{plate.sample_id}_tiles")
                tiling.show_all_rois_global(plate.cropped, tiles, detections_key="colonies",
                                            color=(0, 0, 255), thickness=2,
                                            name=plate.sample_id, output_folder=ext_out)
                tiling.save_plate_tiles_to_csv(ext_out, tiles, name=plate.sample_id)

            # ---- score filtering ----
            timer.start(f"[{img_name}] Score filtering")
            tiling.filter_colonies_by_score(tiles, threshold=score, detections_key="colonies",
                                            score_regression=score_regression)
            timer.stop()

            # ---- saving results ----
            timer.start(f"[{img_name}] Saving results")
            tiling.show_all_rois_global(plate.cropped, tiles, detections_key="colonies",
                                        color=(0, 0, 255), thickness=2,
                                        name=plate.sample_id, output_folder=output_folder)
            tiling.save_plate_tiles_to_csv(output_folder, tiles, name=plate.sample_id)
            timer.stop()
            print(f"Results saved: {plate.sample_id} ({timestamp()})")

            del plate, tiles
            gc.collect()

    except Exception as e:
        print(f"ERROR: {e}")
        raise


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input",  type=str,
                   help="path to the folder with images of plates, or single image")
    p.add_argument("output_folder", type=str,
                   help="place for the counting results")
    p.add_argument("-t", action="store_true",
                   help="use separate method if your plates are transilluminated")
    p.add_argument("-b", action="store_true",
                   help="let program think more to get better precision")

    p.add_argument("--model",      type=str,   default="model/frcnn_lr.onnx",  help=argparse.SUPPRESS)
    p.add_argument("--rows",       type=int,   default=3,                      help=argparse.SUPPRESS)
    p.add_argument("--cols",       type=int,   default=3,                      help=argparse.SUPPRESS)
    p.add_argument("--overlap",    type=float, default=0.1,                    help=argparse.SUPPRESS)
    p.add_argument("--tol",        type=int,   default=3,                      help=argparse.SUPPRESS)
    p.add_argument("--scale",      type=int,   default=512,                    help=argparse.SUPPRESS)
    p.add_argument("--score",      type=float, default=0.20,                   help=argparse.SUPPRESS)
    p.add_argument("--extra",      action="store_true",                        help=argparse.SUPPRESS)
    p.add_argument("--mem-debug",  action="store_true",                        help=argparse.SUPPRESS)
    p.add_argument("--validation", action="store_true",                        help=argparse.SUPPRESS)
    p.add_argument("--ref",        type=str,   default="ref.csv",              help=argparse.SUPPRESS)
    p.add_argument("--no-crop",    action="store_true",                        help=argparse.SUPPRESS)
    p.add_argument("--margin",     type=float, default=1,                      help=argparse.SUPPRESS)
    p.add_argument("--ld",         type=float, default=0.05,                   help=argparse.SUPPRESS)
    p.add_argument("--hd",         type=float, default=-0.10,                  help=argparse.SUPPRESS)
    args = p.parse_args()

    regression = "y = -0.000517x + 0.255172"

    if args.t:
        args.model = "model/frcnn_hr.onnx"
        args.score = 0.20
        regression = "y = -0.000517x + 0.255172"
    if args.b:
        args.rows = 4
        args.cols = 4
        args.score = 0.20
        regression = "y = -0.000517x + 0.255172"

    os.makedirs(args.output_folder, exist_ok=True)
    tee, log_path = setup_logging(args.output_folder)

    # ---- timer & config dict (for report) ----
    timer = StageTimer()

    try:
        print(f"START: {timestamp()}")
        print("=====================================================")
        print("Agar3000 v0.2")
        print("=====================================================")
        print("Configuration:")
        print(f"  Input                   : {args.input}")
        print(f"  Output folder           : {args.output_folder}")
        print(f"  Mode                    : {'Transillumination' if args.t else 'Surface illumination'}")
        print(f"  Model                   : {args.model}")
        if args.no_crop:
            print(f"  No-crop mode        : {args.no_crop}")
        else:
            print(f"  Plate margin offset : {args.margin}")
        print(f"  Grid                    : {args.rows} rows x {args.cols} cols")
        print(f"  Overlap                 : {args.overlap}")
        print(f"  Tolerance               : {args.tol}")
        print(f"  Scale                   : {args.scale}")
        print(f"  Min. score filter       : {args.score}")
        print(f"  Correction regression   : {regression}")
        print(f"  Extra mode              : {args.extra}")
        print("  --- Validation ---")
        print(f"  Validation mode         : {args.validation}")
        print(f"  Reference CSV           : {args.ref}")
        print("=====================================================")
        print(f"Log: {log_path}")
        print("-----------------------------------------------------")

        # resolve image count for the report
        EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        if os.path.isfile(args.input):
            n_images = 1
        else:
            n_images = sum(
                len(glob.glob(os.path.join(args.input, f"*{ext}"))) for ext in EXTS
            )

        config = {
            "Input":               args.input,
            "Output folder":       args.output_folder,
            "Mode":                "Transillumination" if args.t else "Surface illumination",
            "Model":               args.model,
            "Images found":        n_images,
            "Grid":                f"{args.rows} × {args.cols}",
            "Overlap":             args.overlap,
            "Tolerance":           args.tol,
            "Scale":               args.scale,
            "Min. score filter":   args.score,
            "Correction regression": regression,
            "Extra mode":          args.extra,
            "Validation mode":     args.validation,
            "Reference CSV":       args.ref,
            "No-crop mode":        args.no_crop,
            "Plate margin offset": args.margin,
        }

        # ---- main pipeline ----
        run_folder_pipeline(
            input_folder     = args.input,
            output_folder    = args.output_folder,
            model            = args.model,
            grid             = (args.rows, args.cols),
            overlap          = args.overlap,
            tol              = args.tol,
            scale            = args.scale,
            score            = args.score,
            extra            = args.extra,
            mem_debug        = args.mem_debug,
            no_crop          = args.no_crop,
            margin           = args.margin,
            score_regression = (args.ld, args.hd),
            timer            = timer,
        )

        # ---- summarise ----
        timer.start("Colony summarisation")
        summarize_colonies(args.output_folder, f"{args.output_folder}/sum.csv")
        timer.stop()

        # ---- optional validation ----
        if args.validation:
            import subprocess
            timer.start("R validation report")
            cmd = [
                "Rscript", "main/validation.R",
                "-i", f"{args.output_folder}/sum.csv",
                "-o", f"{args.output_folder}/validation_report.html",
                "-r", args.ref,
                "-t", "main/report_template.Rmd",
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"R script failed with exit code {e.returncode}")
                timer.stop()
                sys.exit(1)
            timer.stop()

        print("-----------------------------------------------------")
        print(f"FINISH: {timestamp()}")

        # ---- aggregated timing summary in console ----
        print("=====================================================")
        print("TIMING SUMMARY — aggregated (summed across all plates)")
        print("=====================================================")
        for a in timer.aggregate():
            print(f"  {a['name']:<55} {_fmt(a['elapsed']):>12}")
        print("-----------------------------------------------------")
        print(f"  {'TOTAL':<55} {_fmt(timer.total()):>12}")
        print("=====================================================")

        # ---- CSV + HTML timing report ----
        save_timing_csv(timer, args.output_folder)
        generate_timing_report(timer, args.output_folder, config, log_path)

    finally:
        tee.close()


if __name__ == "__main__":
    main()