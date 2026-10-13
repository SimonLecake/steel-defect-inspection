# Dataset setup

Dataset images and annotations are not redistributed in this repository. Obtain an authorized six-class steel surface defect dataset and prepare:

```text
steel_data/
  dataset.yaml
  train/
    images/        # images
    labels/        # same-stem YOLO .txt annotations
    train.txt      # training image paths
    val.txt        # validation image paths
    test.txt       # held-out image paths
  test/
    images/        # optional competition inference images
```

Each label line is `class_id center_x center_y width height`, with coordinates normalized to [0, 1]. Classes are defined in `dataset.yaml`.

Use absolute image paths in split lists for portability across the training tools. Keep all three splits disjoint and use a representative validation set. The original local example used 1,260 / 14 / 126 images; the 14-image validation set is too small for a reliable production performance claim.

`train_and_predict.py` writes `dataset_local.yaml` for the current machine. For `train_yolo_enhanced.py`, update `dataset.yaml` to the absolute path of your dataset first.

`../convert_annotations.py` demonstrates VOC XML → YOLO conversion. Run it from `yolo-cases/`; on case-sensitive filesystems change its `LABELS` output path to `labels`.

The submission exporter expects numeric image stems (for example, `1400.jpg`). Rename files or adapt the exporter for other naming conventions.
