from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path

import cv2


class ShotClipBuffer:
    """Keeps a rolling frame buffer and saves recent shot clips to disk."""

    def __init__(
        self,
        fps: int,
        frame_size: tuple[int, int],
        pre_seconds: float = 4.0,
        output_dir: str = "replays",
    ):
        self._fps = max(1, int(fps))
        self._frame_size = frame_size  # (w, h)
        max_frames = int(pre_seconds * self._fps)
        self._frames: deque = deque(maxlen=max_frames)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def push(self, frame):
        self._frames.append(frame.copy())

    def save_recent_clip(self, tag: str) -> str | None:
        if not self._frames:
            return None

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        name = f"shot_{tag}_{ts}.mp4"
        out_path = self._output_dir / name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(self._fps), self._frame_size)
        if not writer.isOpened():
            return None

        for frm in self._frames:
            writer.write(frm)
        writer.release()

        return str(out_path)
