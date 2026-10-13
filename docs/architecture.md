# 架构与关键设计

## 单体应用，模块化职责

Gradio 回调直接编排检测、持久化与展示。当前没有独立 REST 服务、消息队列或微服务层，适用于本地演示及早期验证。

| 模块 | 责任 | 关键接口 |
|---|---|---|
| `app.py` | 图片保存、引擎选择、交互与流程编排 | `run_detection()`、`build_app()` |
| `detection_engine.py` | 推理与输出标准化 | `YOLODetector.detect()`、`VLMDetector.detect()` |
| `db_manager.py` | 检测记录与审核状态 | `InspectionDB` |
| `data_export.py` | 提取误检、漏检材料 | `export_bad_cases()`、`export_for_retraining()` |

## 统一结果契约

两种引擎均返回 `DetectionResult`：

```json
{
  "model_type": "yolo",
  "image_path": "temp_uploads/example.jpg",
  "boxes": [{
    "class_id": 3,
    "class_name": "scratches",
    "confidence": 0.87,
    "bbox": [12.0, 34.0, 56.0, 178.0]
  }],
  "vlm_text": "",
  "vlm_defect_types": [],
  "inference_time": 0.123
}
```

以上为格式示例，不是实测预测。坐标统一为原图像素空间中的 `xyxy`。绘框图保留在内存中供界面展示，序列化记录存储原图路径和结构化结果。

YOLO 路线采用本地权重，并包含历史 AAttn 权重格式的兼容补丁。VLM 路线将图片编码后提交给兼容服务，解析 JSON 中的类别、坐标与描述；目前根据模型名处理 Qwen3 的归一化坐标，其他服务需验证坐标约定。

## 记录与审核

SQLite 实际使用一张业务表 `detection_log`。检测框保存在 `detections_json` 字段，没有独立的 `detection_boxes` 表。

审核状态为 `pending`、`correct`、`wrong`、`missed`。这是记录级审核，没有实现逐框修改，也没有审核人、审核历史或模型版本字段。

问题样本导出按 `wrong` / `missed` 选取图片和记录；导出的原始预测不等于修正后的真值。重新标注、合并数据、训练和部署仍为人工步骤。

## 配置与部署

- 检测器首次使用时加载，并在当前进程中复用。
- VLM 支持界面参数、环境变量和默认值。
- 默认监听 `127.0.0.1:7860`；可通过 `GRADIO_SERVER_NAME` 修改。
- 多用户隔离、认证、任务调度、模型注册与生产监控未实现。
- 公共源码不包含数据或权重；运行资源按指南单独准备。
