#!/usr/bin/env python3
"""Check whether a live ZED camera still matches the raw v2 training views.

The checker uses frame 0 from many trajectories as a reference bank. It does not
average those frames: each frame is matched independently, then the geometric
match statistics are aggregated robustly. This avoids blurring the moving arm,
plate, and candy into a misleading reference image.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

CAMERA_INFO = {
    "side": ("exterior_image_1_left", "38651013_left"),
    "wrist": ("wrist_image_left", "15577469_left"),
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", required=True,
                   help="Raw root containing target_*/success|failure/*/traj.hdf5")
    p.add_argument("--camera", choices=CAMERA_INFO, default="side")
    p.add_argument("--reference-frame", type=int, default=0)
    p.add_argument("--max-references", type=int, default=60,
                   help="Deterministic evenly spaced subset; 0 uses every trajectory")
    p.add_argument("--top-fraction", type=float, default=0.70,
                   help="Use only the upper part of the image (default 0.70)")
    p.add_argument("--output-dir", default="/tmp/candy-camera-check")
    p.add_argument("--live-image", default=None,
                   help="Existing RGB/BGR image for an offline test; otherwise capture the ZED directly")
    return p.parse_args()


def find_trajectories(root: Path, limit: int) -> list[Path]:
    # Include both target-specific episodes and arbitrary-target episodes.
    paths = sorted(set(root.glob("target_*/**/traj.hdf5")) |
                   set(root.glob("arbitrary/**/traj.hdf5")))
    if limit and len(paths) > limit:
        paths = [paths[int(i)] for i in np.linspace(0, len(paths) - 1, limit, dtype=int)]
    return paths


def image_key(f: h5py.File, name: str) -> str:
    for key in (f"saved_observation/{name}", f"observation/image/{name}", name):
        if key in f:
            return key
    raise KeyError(f"image {name!r} not found in HDF5 file")


def read_frame(path: Path, name: str, index: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        key = image_key(f, name)
        data = f[key]
        if not 0 <= index < len(data):
            raise IndexError(f"{path}: frame {index} outside 0..{len(data)-1}")
        return np.asarray(data[index])


def capture_live(camera_id: str) -> np.ndarray:
    """Capture one left-ZED frame directly, without env/reset/controller/gripper."""
    try:
        import pyzed.sl as sl
    except ImportError as exc:
        raise SystemExit("pyzed is not installed in this environment") from exc

    try:
        serial = int(camera_id.split("_", 1)[0])
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"Invalid ZED camera id: {camera_id!r}") from exc
    devices = sl.Camera.get_device_list()
    if not any(int(device.serial_number) == serial for device in devices):
        raise SystemExit(
            f"Camera {serial} not found; connected cameras: "
            f"{[int(device.serial_number) for device in devices]}"
        )

    camera = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(serial)
    # Match the mode used by collection/evaluation (the ZED logs report
    # HD1080@15). Comparing HD720 against HD1080 changes the field of view.
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 15
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Could not open ZED {serial}: {status}")

    mat = sl.Mat()
    try:
        for _ in range(30):
            if camera.grab() == sl.ERROR_CODE.SUCCESS:
                camera.retrieve_image(mat, sl.VIEW.LEFT)
                frame = np.asarray(mat.get_data())
                if frame.ndim == 3 and frame.shape[2] >= 3:
                    # ZED returns BGRA; the rest of this script uses RGB.
                    return np.ascontiguousarray(frame[..., :3][..., ::-1])
        raise SystemExit(f"Could not grab a frame from ZED {serial}")
    finally:
        camera.close()


def masked_gray(rgb: np.ndarray, top_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cutoff = max(1, min(gray.shape[0], round(gray.shape[0] * top_fraction)))
    mask[:cutoff, :] = 255
    # The center/lower workspace contains the arm, plate, and candy, which vary
    # between demonstrations and should not decide whether the camera moved.
    x0, x1 = round(.16 * gray.shape[1]), round(.92 * gray.shape[1])
    # Keep the fixed upper scene and narrow side strips; excluding only the
    # central lower 35% avoids the moving arm/plate without starving SIFT.
    y0, y1 = round(.35 * gray.shape[0]), round(.70 * gray.shape[0])
    mask[y0:y1, x0:x1] = 0
    return gray, mask


def estimate(reference: np.ndarray, live: np.ndarray, top_fraction: float) -> dict[str, Any]:
    rg, rm = masked_gray(reference, top_fraction)
    lg, lm = masked_gray(live, top_fraction)
    if hasattr(cv2, "SIFT_create"):
        detector, norm = cv2.SIFT_create(nfeatures=1600), cv2.NORM_L2
    else:
        detector, norm = cv2.ORB_create(nfeatures=2200), cv2.NORM_HAMMING
    kp_r, des_r = detector.detectAndCompute(rg, rm)
    kp_l, des_l = detector.detectAndCompute(lg, lm)
    result: dict[str, Any] = {
        "reference_keypoints": 0 if kp_r is None else len(kp_r),
        "live_keypoints": 0 if kp_l is None else len(kp_l),
        "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0,
        "translation_px": None, "rotation_deg": None, "scale": None,
        "median_error_px": None,
    }
    if des_r is None or des_l is None or len(des_r) < 2 or len(des_l) < 2:
        return result
    pairs = cv2.BFMatcher(norm).knnMatch(des_r, des_l, k=2)
    good = [m for m, n in pairs if m.distance < .75 * n.distance]
    result["good_matches"] = len(good)
    if len(good) < 4:
        return result
    src = np.float32([kp_r[m.queryIdx].pt for m in good])
    dst = np.float32([kp_l[m.trainIdx].pt for m in good])
    affine, inlier = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
        maxIters=3000, confidence=.995)
    if affine is None or inlier is None:
        return result
    keep = inlier.ravel().astype(bool)
    result["inliers"] = int(keep.sum())
    result["inlier_ratio"] = float(keep.mean())
    projected = cv2.transform(src[None], affine)[0]
    err = np.linalg.norm(projected - dst, axis=1)
    result["median_error_px"] = float(np.median(err[keep])) if keep.any() else None
    a, b = float(affine[0, 0]), float(affine[1, 0])
    result["scale"] = float(math.sqrt(a * a + b * b))
    result["rotation_deg"] = float(math.degrees(math.atan2(b, a)))
    result["translation_px"] = float(np.linalg.norm(affine[:, 2]))
    result["_keypoints"] = (kp_r, kp_l, good, keep)
    return result


def save_match_image(reference: np.ndarray, live: np.ndarray, result: dict[str, Any], out: Path) -> None:
    data = result.get("_keypoints")
    if data is None:
        cv2.imwrite(str(out), cv2.cvtColor(np.hstack((reference, live)), cv2.COLOR_RGB2BGR))
        return
    kp_r, kp_l, good, keep = data
    canvas = cv2.drawMatches(
        reference, kp_r, live, kp_l, good, None,
        matchesMask=keep.astype(np.uint8).tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def median(results: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in results if r.get(key) is not None]
    return float(np.median(values)) if values else None


def main() -> None:
    a = args()
    root = Path(a.reference_root).expanduser()
    out = Path(a.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    image_name, camera_id = CAMERA_INFO[a.camera]
    paths = find_trajectories(root, a.max_references)
    if not paths:
        raise SystemExit(f"No trajectory files below {root}")
    references: list[tuple[Path, np.ndarray]] = []
    for path in paths:
        try:
            references.append((path, read_frame(path, image_name, a.reference_frame)))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    if not references:
        raise SystemExit("No readable reference frames")

    if a.live_image:
        bgr = cv2.imread(a.live_image)
        if bgr is None:
            raise SystemExit(f"Cannot read {a.live_image}")
        live = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        print(f"Capturing {a.camera} ({camera_id}) without reset or gripper command...")
        live = capture_live(camera_id)
    if live.shape[:2] != references[0][1].shape[:2]:
        live = cv2.resize(live, (references[0][1].shape[1], references[0][1].shape[0]))
    cv2.imwrite(str(out / "live.png"), cv2.cvtColor(live, cv2.COLOR_RGB2BGR))

    results = []
    for path, ref in references:
        r = estimate(ref, live, a.top_fraction)
        r.pop("_keypoints", None)
        r["reference"] = str(path)
        results.append(r)
    valid = [r for r in results if r["inliers"] >= 4]
    if not valid:
        (out / "report.json").write_text(json.dumps({"results": results}, indent=2))
        raise SystemExit("No robust feature matches. Inspect live.png and check the camera/lighting.")
    best = max(valid, key=lambda r: r["inliers"])
    best_path = Path(best["reference"])
    best_ref = next(ref for path, ref in references if path == best_path)
    # Recompute for match drawing because the compact report omits keypoint objects.
    best_full = estimate(best_ref, live, a.top_fraction)
    cv2.imwrite(str(out / "best_reference.png"), cv2.cvtColor(best_ref, cv2.COLOR_RGB2BGR))
    save_match_image(best_ref, live, best_full, out / "matches.png")

    report = {
        "camera": a.camera,
        "camera_id": camera_id,
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
