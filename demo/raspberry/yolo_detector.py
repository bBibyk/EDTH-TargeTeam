from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class YoloDetector:
    def __init__(
        self,
        model_path: str | Path,
        confidence: float,
        iou: float,
        max_det: int,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO  # type: ignore[import-not-found]

        self._model = YOLO(str(model_path))
        self._confidence = confidence
        self._iou = iou
        self._max_det = max_det
        self._device = device

    def infer(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        results = self._model.predict(
            frame_bgr,
            conf=self._confidence,
            iou=self._iou,
            max_det=self._max_det,
            device=self._device,
            verbose=False,
        )
        detections: list[dict[str, Any]] = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        for i in range(len(xyxy)):
            cls_id = int(clss[i])
            label = str(names.get(cls_id, cls_id))
            detections.append(
                {
                    "xyxy": [float(v) for v in xyxy[i].tolist()],
                    "conf": float(confs[i]),
                    "cls": cls_id,
                    "label": label,
                }
            )
        return detections
