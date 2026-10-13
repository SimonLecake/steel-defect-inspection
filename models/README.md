# Model weights

Place your trained six-class steel-defect checkpoint at `models/best.pt`.
Weights are not included in Git. The app loads the model lazily on the first YOLO request.

Class order must match `detection_engine.py` and `yolo-cases/steel_data/dataset.yaml`:
`crazing`, `inclusion`, `pitted_surface`, `scratches`, `patches`, `rolled-in_scale`.

A general-purpose COCO checkpoint is not a trained steel-defect detector. Train or obtain a compatible checkpoint first. After replacing a loaded checkpoint, restart the app.

The VLM route does not require a local checkpoint; configure your own compatible API endpoint and key.
