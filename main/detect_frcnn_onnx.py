# coding: ascii
# detect_frcnn_onnx.py - Single Image Detection for MMDeploy FasterRCNN ONNX model.
#
# CLI usage:
#   python detect_frcnn_onnx.py --model end2end.onnx --image demo.png
#   python detect_frcnn_onnx.py --model end2end.onnx --image demo.png --no-normalize
#   python detect_frcnn_onnx.py --model end2end.onnx --image demo.png --conf 0.1 --save
#   python detect_frcnn_onnx.py --model end2end.onnx --image demo.png --no-rgb
#
# API usage:
#   from detect_frcnn_onnx import load_model, detect, detect_on_tiles
#
#   model = load_model("end2end.onnx")
#
#   # single image (path or numpy array)
#   detections, img_bgr = detect(model, "demo.png", size=512, conf=0.25)
#
#   # tiled inference
#   tiles = detect_on_tiles(tiles, model, size=2048, conf=0.25)
#
#   # detections: list of dicts {x1, y1, x2, y2, score, label}
#
# Requirements: onnxruntime or onnxruntime-gpu (auto GPU->CPU fallback)

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ImageNet/COCO mean and std in RGB order (pixel scale 0-255)
MEAN = np.array([123.675, 116.28,  103.53], dtype=np.float32)
STD  = np.array([58.395,  57.12,   57.375], dtype=np.float32)

MIN_BOX_SIZE = 4  # drop boxes smaller than this in either dimension (original coords)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(model_path):
    """
    Load an ONNX model with automatic GPU->CPU fallback.
    Returns an onnxruntime.InferenceSession.
    CUDA_VISIBLE_DEVICES is set before session creation to prevent ORT from
    spawning affinity-failing threads on nodes with broken/missing CUDA drivers.
    """
    import subprocess

    # Check if GPU is actually usable
    gpu_ok = False
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        gpu_ok = True
    except Exception:
        pass

    # If GPU is not usable, hide CUDA from ORT entirely.
    # This must be done before InferenceSession is created.
    # If onnxruntime was already imported, this only works if the CUDA
    # provider has not been initialized yet -- which is the case here
    # since we call this once at startup.
    if not gpu_ok:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu_ok \
                else ["CPUExecutionProvider"]

    available = [p.split("ExecutionProvider")[0] for p in ort.get_available_providers()]
    active    = [p for p in providers if p in ort.get_available_providers()]
    device    = "GPU" if "CUDAExecutionProvider" in active else "CPU"

    print("[INFO] Available ORT providers : {}".format(available))
    print("[INFO] Loading model           : {}".format(model_path))
    print("[INFO] Active device           : {}".format(device))

    n_threads = int(
        os.environ.get("SLURM_CPUS_PER_TASK") or
        os.environ.get("SLURM_CPUS_ON_NODE")  or
        os.cpu_count()                          or
        1
    )
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = n_threads
    print("[INFO] ORT threads             : {}".format(n_threads))

    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)

    print("[INFO] Session provider        : {}".format(
        session.get_providers()[0].replace("ExecutionProvider", "")
    ))
    return session

# --------------------------------------------------------------------------- #
# Pre-processing  (identical to TorchScript version, minus .to(DEVICE))
# --------------------------------------------------------------------------- #

def preprocess(image, size, normalize=True, rgb=True):
    """
    Load and prepare a single image for inference.

    Args:
        image     : str, Path, or np.ndarray (BGR, HxWxC)
        size      : int  - square input size for the model
        normalize : bool - apply ImageNet mean/std normalisation (default True)
        rgb       : bool - convert BGR->RGB before processing (default True)

    Returns:
        tensor  : np.ndarray (1, 3, size, size) float32 - ready for ORT
        img_bgr : np.ndarray original BGR image (for visualisation)
        orig_w  : int
        orig_h  : int
        scale   : float  resize scale factor (to map boxes back)
    """
    if isinstance(image, np.ndarray):
        img_bgr = image
    else:
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise FileNotFoundError("Could not read image: {}".format(image))

    orig_h, orig_w = img_bgr.shape[:2]

    if rgb:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    else:
        img = img_bgr.astype(np.float32)

    # Keep-ratio resize then zero-pad to size x size
    scale = size / max(orig_h, orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    img   = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[:new_h, :new_w] = img

    if normalize:
        canvas = (canvas - MEAN) / STD

    # ORT expects (1, C, H, W) numpy float32 - no .to(device) needed
    tensor = canvas.transpose(2, 0, 1)[np.newaxis, ...]   # (1, 3, H, W)
    return tensor, img_bgr, orig_w, orig_h, scale


# --------------------------------------------------------------------------- #
# Post-processing  (same logic, adapted for numpy outputs)
# --------------------------------------------------------------------------- #

def _filter_small_boxes(detections, min_size):
    """Remove boxes whose width or height is below min_size pixels."""
    kept    = [d for d in detections
               if (d["x2"] - d["x1"]) >= min_size and (d["y2"] - d["y1"]) >= min_size]
    removed = len(detections) - len(kept)
    if removed:
        print("[INFO] Removed {} box(es) smaller than {}x{} px.".format(
            removed, min_size, min_size))
    return kept


def postprocess(outputs, conf, scale, min_box_size=MIN_BOX_SIZE):
    """
    Parse raw ORT output into a list of detection dicts.

    MMDeploy ONNX FasterRCNN typically returns two outputs:
        outputs[0] : (1, N, 5)  - [x1, y1, x2, y2, score]
        outputs[1] : (1, N)     - class labels

    Args:
        outputs      : list of np.ndarray returned by session.run()
        conf         : float - confidence threshold
        scale        : float - resize scale to map boxes to original coords
        min_box_size : int   - minimum box side length in original image px

    Returns:
        List of dicts: {x1, y1, x2, y2, score, label}
    """
    boxes_scores = outputs[0][0]           # (N, 5)
    labels_raw   = outputs[1][0] if len(outputs) > 1 else None   # (N,)

    if len(boxes_scores) >= 100:
        print("[WARN] Hit max_per_img=100 cap - detections may be incomplete.")
        print("[WARN] Re-export with higher keep_top_k to fix this.")

    detections = []
    for i, det in enumerate(boxes_scores):
        x1, y1, x2, y2, score = det.tolist()
        if score < conf:
            continue
        label = int(labels_raw[i]) if labels_raw is not None else -1
        detections.append({
            "x1":    x1 / scale,
            "y1":    y1 / scale,
            "x2":    x2 / scale,
            "y2":    y2 / scale,
            "score": score,
            "label": label,
        })

    detections = _filter_small_boxes(detections, min_box_size)
    return detections


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def detect(session, image, size=512, conf=0.25,
           normalize=True, rgb=True, min_box_size=MIN_BOX_SIZE, verbose=True):
    """
    Run detection on a single image.

    Args:
        session      : ORT InferenceSession (from load_model())
        image        : str, Path, or np.ndarray (BGR)
        size         : int   - square input size (default 512)
        conf         : float - confidence threshold (default 0.25)
        normalize    : bool  - apply ImageNet mean/std (default True)
        rgb          : bool  - convert BGR->RGB (default True)
        min_box_size : int   - drop boxes smaller than this (default 4)
        verbose      : bool  - print per-image info (default True)

    Returns:
        detections : list of dicts {x1, y1, x2, y2, score, label}
        img_bgr    : np.ndarray original BGR image
    """
    tensor, img_bgr, orig_w, orig_h, scale = preprocess(
        image, size, normalize=normalize, rgb=rgb
    )

    if verbose:
        norm_str = "normalised" if normalize else "not normalised"
        rgb_str  = "RGB" if rgb else "BGR"
        print("[INFO] {}x{}  scale={:.4f}  {}  {}  tensor:{}".format(
            orig_w, orig_h, scale, norm_str, rgb_str, tensor.shape))

    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: tensor})

    detections = postprocess(outputs, conf, scale, min_box_size)

    if verbose:
        print("[INFO] Kept {} detection(s) (conf>={}, min_box={}px).".format(
            len(detections), conf, min_box_size))
        if not detections:
            print("[WARN] No detections. Try lowering --conf.")

    return detections, img_bgr


def detect_on_tiles(tiles, session, size=512, conf=0.25,
                    normalize=True, rgb=True, min_box_size=MIN_BOX_SIZE,
                    out_key="colonies"):
    """
    Run detection on each tile and map results back to original image coords.
    Identical contract to the TorchScript version - only 'model' -> 'session'.
    """
    total = len(tiles)
    for i, (rc, d) in enumerate(tiles.items()):
        tile             = d["tile"]
        x0, y0, x1_bbox, y1_bbox = d["bbox"]
        print("[tile] {}/{}  rc={})".format(
            i + 1, total, rc), end="")

        detections, _ = detect(
            session, tile,
            size=size, conf=conf,
            normalize=normalize, rgb=rgb,
            min_box_size=min_box_size,
            verbose=False,
        )

        dets = []
        for det in detections:
            tx1, ty1, tx2, ty2 = det["x1"], det["y1"], det["x2"], det["y2"]
            roi        = (int(ty1), int(tx1), int(ty2), int(tx2))
            roi_global = (roi[0] + y0, roi[1] + x0,
                          roi[2] + y0, roi[3] + x0)
            dets.append({
                "roi":        roi,
                "roi_global": roi_global,
                "score":      det["score"],
                "label":      det["label"],
            })

        print(" -> {} detection(s)".format(len(dets)))
        d[out_key] = dets

    return tiles


# --------------------------------------------------------------------------- #
# Visualisation  (unchanged)
# --------------------------------------------------------------------------- #

def draw_boxes(img_bgr, detections, color=(0, 0, 255), use_global=False):
    out = img_bgr.copy()
    for d in detections:
        if "x1" in d:
            x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        elif use_global:
            y1, x1, y2, x2 = d["roi_global"]
        else:
            y1, x1, y2, x2 = d["roi"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, "{:.2f}".format(d["score"]), (x1 + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="MMDeploy FasterRCNN ONNX inference")
    parser.add_argument("--model",        required=True,            help="Path to ONNX model (.onnx)")
    parser.add_argument("--image",        required=True,            help="Path to input image")
    parser.add_argument("--size",         type=int,   default=512,  help="Model input size (default: 512)")
    parser.add_argument("--conf",         type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save",         action="store_true",      help="Save annotated output to current folder")
    parser.add_argument("--no-normalize", action="store_true",      help="Disable ImageNet mean/std normalisation")
    parser.add_argument("--no-rgb",       action="store_true",      help="Skip BGR->RGB conversion")
    parser.add_argument("--min-box",      type=int, default=MIN_BOX_SIZE,
                        help="Minimum box side length in px (default: {})".format(MIN_BOX_SIZE))
    args = parser.parse_args()

    if not Path(args.model).exists():
        print("[ERROR] Model not found: {}".format(args.model)); sys.exit(1)
    if not Path(args.image).exists():
        print("[ERROR] Image not found: {}".format(args.image)); sys.exit(1)

    session = load_model(args.model)

    normalize = not args.no_normalize
    rgb       = not args.no_rgb
    print("[INFO] normalize={}  rgb={}".format(normalize, rgb))

    detections, img_bgr = detect(
        session, args.image,
        size=args.size,
        conf=args.conf,
        normalize=normalize,
        rgb=rgb,
        min_box_size=args.min_box,
    )

    for i, d in enumerate(detections):
        print("  det {:3d}: score={:.4f}  label={}  bbox=[{:.1f}, {:.1f}, {:.1f}, {:.1f}]".format(
            i, d["score"], d["label"], d["x1"], d["y1"], d["x2"], d["y2"]))

    if args.save:
        out_path = "output_{}".format(Path(args.image).name)
        cv2.imwrite(out_path, draw_boxes(img_bgr, detections))
        print("[INFO] Saved -> {}".format(os.path.abspath(out_path)))

    print("\n[DONE]")


if __name__ == "__main__":
    main()