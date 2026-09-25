#!/usr/bin/env python3
"""Compare a live Egg-task ZED view with frames from collected Egg data.

This is intentionally separate from compare_camera_view.py, which is Candy-only.
It does not reset the robot, start a controller, or command the gripper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from scripts.compare_camera_view import (
    CAMERA_INFO,
    capture_live,
    estimate,
    masked_gray,
    median,
    save_match_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference-root",
        default="/iris/u/ronpo/expo-ft-data/egg_v1",
        help="Egg dataset root containing card_black/card_white/{success,failure}/...",
    )
    p.add_argument("--card", choices=("all", "black", "white"), default="all")
    p.add_argument("--camera", choices=CAMERA_INFO, default="wrist")
    p.add_argument("--reference-frame", type=int, default=0)
    p.add_argument("--max-references", type=int, default=60)
    p.add_argument("--top-fraction", type=float, default=0.70)
    p.add_argument("--output-dir", default="/tmp/egg-camera-check")
    p.add_argument("--live-image", default=None,
                   help="Use an existing image instead of opening the ZED")
    return p.parse_args()


def find_trajectories(root: Path, card: str, limit: int) -> list[Path]:
    cards = ("card_black", "card_white") if card == "all" else (f"card_{card}",)
    paths = []
    for card_name in cards:
        paths.extend(root.glob(f"{card_name}/success/*/traj.hdf5"))
        paths.extend(root.glob(f"{card_name}/failure/*/traj.hdf5"))
    paths = sorted(set(paths))
    if limit and len(paths) > limit:
        paths = [paths[int(i)] for i in np.linspace(0, len(paths) - 1, limit, dtype=int)]
    return paths


def read_frame(path: Path, name: str, index: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        candidates = (f"saved_observation/{name}", f"observation/image/{name}", name)
        key = next((key for key in candidates if key in f), None)
        if key is None:
            raise KeyError(f"image {name!r} not found")
        data = f[key]
        if not 0 <= index < len(data):
            raise IndexError(f"frame {index} outside 0..{len(data)-1}")
        return np.asarray(data[index])


def main() -> None:
    a = parse_args()
    root = Path(a.reference_root).expanduser()
    out = Path(a.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    image_name, camera_id = CAMERA_INFO[a.camera]
    paths = find_trajectories(root, a.card, a.max_references)
    if not paths:
        raise SystemExit(f"No Egg trajectories found below {root} for card={a.card}")

    references = []
    for path in paths:
        try:
            references.append((path, read_frame(path, image_name, a.reference_frame)))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    if not references:
        raise SystemExit("No readable Egg reference frames")

    if a.live_image:
        bgr = cv2.imread(a.live_image)
        if bgr is None:
            raise SystemExit(f"Cannot read {a.live_image}")
        live = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        print(f"Capturing {a.camera} ({camera_id}) without reset/controller/gripper...")
        live = capture_live(camera_id)
    if live.shape[:2] != references[0][1].shape[:2]:
        live = cv2.resize(live, (references[0][1].shape[1], references[0][1].shape[0]))
    cv2.imwrite(str(out / "live.png"), cv2.cvtColor(live, cv2.COLOR_RGB2BGR))

    results = []
    for path, ref in references:
        result = estimate(ref, live, a.top_fraction)
        result.pop("_keypoints", None)
        result["reference"] = str(path)
        results.append(result)
    valid = [r for r in results if r["inliers"] >= 4]
    if not valid:
        (out / "report.json").write_text(json.dumps({"results": results}, indent=2))
        raise SystemExit("No robust feature matches; inspect live.png and check USB/lighting.")

    best = max(valid, key=lambda r: r["inliers"])
    best_path = Path(best["reference"])
    best_ref = next(ref for path, ref in references if path == best_path)
    best_full = estimate(best_ref, live, a.top_fraction)
    cv2.imwrite(str(out / "best_reference.png"), cv2.cvtColor(best_ref, cv2.COLOR_RGB2BGR))
    save_match_image(best_ref, live, best_full, out / "matches.png")

    report = {
        "task": "egg",
        "camera": a.camera,
        "camera_id": camera_id,
        "card_filter": a.card,
        "reference_root": str(root),
        "reference_frame": a.reference_frame,
        "references_used": len(references),
        "valid_alignments": len(valid),
        "median_inliers": median(valid, "inliers"),
        "median_inlier_ratio": median(valid, "inlier_ratio"),
        "median_translation_px": median(valid, "translation_px"),
        "median_rotation_deg": median(valid, "rotation_deg"),
        "median_scale": median(valid, "scale"),
        "median_reprojection_error_px": median(valid, "median_error_px"),
        "best_reference": str(best_path),
        "all_results": results,
    }
    report["needs_alignment_review"] = bool(
        (report["median_inlier_ratio"] is not None and report["median_inlier_ratio"] < .45)
        or (report["median_reprojection_error_px"] is not None and report["median_reprojection_error_px"] > 3)
        or (report["median_translation_px"] is not None and report["median_translation_px"] > 8)
        or (report["median_rotation_deg"] is not None and abs(report["median_rotation_deg"]) > 2)
        or (report["median_scale"] is not None and abs(report["median_scale"] - 1) > .03)
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "all_results"}, indent=2))
    print(f"Wrote {out}/live.png, best_reference.png, matches.png, report.json")


if __name__ == "__main__":
    main()
