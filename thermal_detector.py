
from ultralytics import YOLO
from pathlib import Path

class ThermalAnimalDetector:
    def __init__(self, model_path, confidence=0.25):
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)

        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, image_path):
        if not Path(image_path).exists():
            raise FileNotFoundError(image_path)

        results = self.model.predict(
            source=str(image_path),
            imgsz=640,
            conf=self.confidence,
            verbose=False
        )

        result = results[0]
        detections = []

        for box in result.boxes:
            detections.append({
                "class": "animal",
                "confidence": round(float(box.conf.item()), 4),
                "bbox": [
                    round(float(v), 2)
                    for v in box.xyxy[0].tolist()
                ]
            })

        return {
            "image": str(image_path),
            "animal_count": len(detections),
            "detections": detections
        }
