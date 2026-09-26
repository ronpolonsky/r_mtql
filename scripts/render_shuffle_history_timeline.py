#!/usr/bin/env python3
"""Render one Shuffle episode as a training-history timeline.

This is read-only with respect to the source HDF5.  Each output video frame
shows the current observation at anchor ``t`` and the exact H14/S6 history
slots produced by ``build_shuffle_history_indices``.  Source-frame labels make
cue retention and the first-motion timing directly inspectable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.mtql_shuffle_device_cache import build_shuffle_history_indices


CAMERA = "exterior_image_1_left"
BG = (24, 26, 31)
WHITE = (238, 240, 244)
MUTED = (165, 170, 180)
ACCENT = (80, 190, 255)
CUE = (90, 210, 125)
PRE = (245, 190, 70)
MOTION = (255, 105, 105)


def font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def first_sustained_motion(actions: np.ndarray) -> int:
    active = np.any(np.abs(np.asarray(actions[:, :6])) > 1e-4, axis=1)
    starts = np.flatnonzero(active[:-1] & active[1:])
    if not len(starts):
        raise ValueError("Trajectory has no two-frame sustained arm motion")
    return int(starts[0])


def fit_image(frame: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (8, 9, 12))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def label_box(draw: ImageDraw.ImageDraw, xy, text: str, fill, text_fill=BG):
    x, y = xy
    box = draw.textbbox((x, y), text, font=font(16))
    draw.rounded_rectangle(
        (box[0] - 5, box[1] - 3, box[2] + 5, box[3] + 3),
        radius=4,
        fill=fill,
    )
    draw.text((x, y), text, font=font(16), fill=text_fill)


def render_frame(
    images: np.ndarray,
    actions: np.ndarray,
    history_indices: np.ndarray,
    history_padding: np.ndarray,
    anchor: int,
    motion_start: int,
    hist_length: int,
    hist_stride: int,
    cue_frames: int,
) -> np.ndarray:
    width, height = 1440, 900
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    title_font = font(25)
    body_font = font(17)
    small_font = font(14)

    draw.text(
        (24, 16),
        f"Shuffle v2 edited7  |  H{hist_length}/S{hist_stride}  |  anchor t={anchor}",
        font=title_font,
        fill=WHITE,
    )
    if anchor < motion_start:
        phase = "cue / pre-motion"
        phase_color = PRE
    elif anchor == motion_start:
        phase = "FIRST SUSTAINED MOTION"
        phase_color = MOTION
    else:
        phase = "moving trajectory"
        phase_color = MOTION
    label_box(draw, (24, 52), phase, phase_color)
    draw.text(
        (360, 56),
        f"cue frames: 0–{cue_frames - 1}   |   pre-motion frame: {cue_frames}   |   motion starts: {motion_start}",
        font=body_font,
        fill=MUTED,
    )

    current_x, current_y, current_w, current_h = 24, 100, 320, 180
    draw.text((current_x, current_y - 25), "current observation (not a history slot)", font=body_font, fill=ACCENT)
    current = fit_image(images[anchor], current_w, current_h)
    canvas.paste(current, (current_x, current_y))
    current_color = MOTION if anchor >= motion_start else PRE
    draw.rectangle((current_x, current_y, current_x + current_w - 1, current_y + current_h - 1), outline=current_color, width=4)

    action = actions[anchor, :6]
    axis = int(np.argmax(np.abs(action[:3])))
    axis_name = ("x", "y", "z")[axis]
    axis_sign = "+" if action[axis] >= 0 else "−"
    draw.text((24, 298), "action at t", font=body_font, fill=ACCENT)
    draw.text((24, 324), np.array2string(action, precision=3, suppress_small=True), font=small_font, fill=WHITE)
    draw.text((24, 350), f"dominant translational component: {axis_name}{axis_sign}", font=body_font, fill=WHITE)

    # History slots, oldest to newest, in the same order passed to the model.
    tile_w, tile_h = 150, 84
    x0, y0, gap_x, gap_y = 350, 100, 5, 34
    draw.text((x0, y0 - 25), "history slots passed to the model (oldest → newest)", font=body_font, fill=ACCENT)
    indices = history_indices[anchor]
    padding = history_padding[anchor]
    for slot, (source_index, is_padding) in enumerate(zip(indices, padding)):
        row, col = divmod(slot, 7)
        x = x0 + col * (tile_w + gap_x)
        y = y0 + row * (tile_h + gap_y)
        image = fit_image(images[int(source_index)], tile_w, tile_h)
        canvas.paste(image, (x, y))
        local = int(source_index)
        if is_padding:
            border = (150, 150, 155)
            text = f"h{slot}: t{local} (pad)"
        elif local < cue_frames:
            border = CUE
            text = f"h{slot}: t{local} CUE"
        elif local == cue_frames:
            border = PRE
            text = f"h{slot}: t{local} PRE"
        elif local >= motion_start:
            border = MOTION
            text = f"h{slot}: t{local} move"
        else:
            border = WHITE
            text = f"h{slot}: t{local}"
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=border, width=3)
        draw.rectangle((x + 3, y + 3, x + tile_w - 4, y + 22), fill=(0, 0, 0))
        draw.text((x + 7, y + 5), text, font=small_font, fill=border)

    # Compact action timeline: translational x/y/z, with current anchor and
    # first sustained motion marked. This is a visual aid, not a new label.
    plot_x, plot_y, plot_w, plot_h = 24, 450, 1390, 365
    draw.text((plot_x, plot_y - 27), "executed translational action over the episode", font=body_font, fill=ACCENT)
    draw.rectangle((plot_x, plot_y, plot_x + plot_w, plot_y + plot_h), outline=(75, 80, 90), width=1)
    values = actions[:, :3]
    max_abs = max(float(np.max(np.abs(values))), 1e-6)
    zero_y = plot_y + plot_h // 2
    draw.line((plot_x, zero_y, plot_x + plot_w, zero_y), fill=(85, 90, 100), width=1)
    colors = ((255, 130, 130), (120, 220, 150), (120, 175, 255))
    for dim, color in enumerate(colors):
        points = []
        for i, value in enumerate(values[:, dim]):
            px = plot_x + int(i * plot_w / max(1, len(values) - 1))
            py = zero_y - int(float(value) / max_abs * (plot_h * 0.43))
            points.append((px, py))
        draw.line(points, fill=color, width=3)
    current_px = plot_x + int(anchor * plot_w / max(1, len(values) - 1))
    motion_px = plot_x + int(motion_start * plot_w / max(1, len(values) - 1))
    draw.line((motion_px, plot_y, motion_px, plot_y + plot_h), fill=MOTION, width=2)
    draw.line((current_px, plot_y, current_px, plot_y + plot_h), fill=ACCENT, width=3)
    draw.text((plot_x + 8, plot_y + 8), "x", font=small_font, fill=colors[0])
    draw.text((plot_x + 35, plot_y + 8), "y", font=small_font, fill=colors[1])
    draw.text((plot_x + 62, plot_y + 8), "z", font=small_font, fill=colors[2])
    draw.text((motion_px + 5, plot_y + 8), "first motion", font=small_font, fill=MOTION)
    draw.text((current_px + 5, plot_y + plot_h - 26), f"t={anchor}", font=small_font, fill=ACCENT)
    draw.text((plot_x, plot_y + plot_h + 8), "0", font=small_font, fill=MUTED)
    draw.text((plot_x + plot_w - 30, plot_y + plot_h + 8), str(len(values) - 1), font=small_font, fill=MUTED)

    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_hdf5", type=Path)
    parser.add_argument("output_mp4", type=Path)
    parser.add_argument("--hist-length", type=int, default=14)
    parser.add_argument("--hist-stride", type=int, default=6)
    parser.add_argument("--cue-frames", type=int, default=7)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--camera", default=CAMERA)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    with h5py.File(args.traj_hdf5, "r") as handle:
        images = np.asarray(handle[f"saved_observation/{args.camera}"], dtype=np.uint8)
        actions = np.asarray(handle["action/executed_action"], dtype=np.float32)

    if args.hist_length < args.cue_frames:
        raise ValueError("hist-length must be at least cue-frames")
    motion_start = first_sustained_motion(actions)
    history_indices, history_padding = build_shuffle_history_indices(
        np.asarray([0]),
        np.asarray([len(actions) - 1]),
        size=len(actions),
        hist_length=args.hist_length,
        hist_stride=args.hist_stride,
        cue_frames=args.cue_frames,
    )
    start = max(0, args.start)
    end = len(actions) - 1 if args.end is None else min(args.end, len(actions) - 1)
    if start > end:
        raise ValueError("start must be <= end")
    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        args.output_mp4,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for anchor in range(start, end + 1):
            writer.append_data(
                render_frame(
                    images,
                    actions,
                    history_indices,
                    history_padding,
                    anchor,
                    motion_start,
                    args.hist_length,
                    args.hist_stride,
                    args.cue_frames,
                )
            )
    print(f"wrote {args.output_mp4} for t={start}..{end}; motion_start={motion_start}")
    print(f"history at motion start: {history_indices[motion_start].tolist()}")


if __name__ == "__main__":
    main()
