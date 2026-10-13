"""Offline contract checks; no checkpoint, dataset, or API key required."""
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from detection_engine import DetectionBox, DetectionResult, VLMDetector
from db_manager import InspectionDB
from data_export import export_bad_cases, export_for_retraining


class WorkflowTests(unittest.TestCase):
    def test_qwen3_coordinates_and_description(self):
        detector = VLMDetector.__new__(VLMDetector)
        detector.model_name = "qwen3-vl-plus"
        reply = '```json\n[{"bbox_2d":[100,200,500,800],"label":"scratches","description":"linear mark"}]\n```'
        boxes, classes, text = detector._parse_bbox_response(reply, 400, 200)
        self.assertEqual((boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2), (40, 40, 200, 160))
        self.assertEqual(boxes[0].class_id, 3)
        self.assertEqual(classes, ["scratches"])
        self.assertIn("linear mark", text)

    def test_absolute_coordinates_empty_and_invalid_response(self):
        detector = VLMDetector.__new__(VLMDetector)
        detector.model_name = "qwen2.5-vl"
        boxes, _, _ = detector._parse_bbox_response('[{"bbox_2d":[10,20,50,80],"label":"patches"}]', 400, 200)
        self.assertEqual((boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2), (10,20,50,80))
        self.assertEqual(detector._parse_bbox_response('[]', 400, 200)[0], [])
        with self.assertRaises(json.JSONDecodeError):
            detector._parse_bbox_response('not JSON', 400, 200)

    def test_review_persistence_statistics_and_export_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = str(root / "inspection.db")
            db = InspectionDB(db_path)
            ids = []
            for index, state in enumerate(("correct", "wrong", "missed", "pending")):
                picture = root / f"{index}.png"
                Image.new("RGB", (100,100), "gray").save(picture)
                result = DetectionResult("yolo", str(picture), boxes=[DetectionBox(3,"scratches",0.8,10,10,30,50)])
                record_id = db.save_detection(result)
                db.update_review(record_id, state, "test fixture")
                ids.append(record_id)
            reopened = InspectionDB(db_path)
            self.assertEqual(reopened.get_record_by_id(ids[1])["review_status"], "wrong")
            stats = reopened.get_statistics()
            self.assertEqual(stats["total"], 4)
            self.assertEqual(stats["reviewed_count"], 3)
            self.assertEqual(stats["accuracy"], 33.3)
            destination, count = export_bad_cases(db_path, str(root / "bad"))
            self.assertEqual(count, 2)
            self.assertEqual({p.name for p in (Path(destination)/"images").iterdir()}, {"1.png","2.png"})
            info = json.loads((Path(destination)/"export_info.json").read_text(encoding="utf-8"))
            self.assertEqual((info["wrong_count"],info["missed_count"]),(1,1))
            destination, count = export_for_retraining(db_path, str(root / "retrain"))
            self.assertEqual(count, 2)
            self.assertTrue((Path(destination)/"review_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
