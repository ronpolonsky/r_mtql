#!/usr/bin/env python3
"""Render a compact two-camera MP4 from one saved episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    with np.load(args.episode) as d:
        agent = d["obs_image"]
        wrist = d["obs_wrist_image"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    h, w = agent.shape[1:3]
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w * 2, h)
    )
    frame_indices = list(range(0, len(agent), args.frame_stride))
    if frame_indices[-1] != len(agent) - 1:
        frame_indices.append(len(agent) - 1)
    for i in frame_indices:
        frame = np.concatenate([agent[i], wrist[i]], axis=1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame, (0, 0), (w * 2, 20), (0, 0, 0), -1)
        status = args.label
        if i + args.frame_stride >= len(agent) and args.label.startswith("TARGET"):
            status = "SUCCESS: TARGET DRAWER OPEN"
        elif args.label.startswith("TARGET"):
            status = "SEARCHING"
        cv2.putText(
            frame, f"{status} | transition {i}/{len(agent)-1}", (5, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    main()
