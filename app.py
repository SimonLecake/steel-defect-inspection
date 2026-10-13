# -*- coding: utf-8 -*-
"""
AI质检工作台 - Gradio 界面
功能:
  Tab1 - 缺陷检测: 上传图片, 选择模型(YOLO/VLM), 查看检测结果
  Tab2 - 检测记录: 查看历史记录, 人工审核(正确/误检/漏检)
  Tab3 - 统计分析: 检测数量、准确率、缺陷类别分布
  Tab4 - 数据导出: 导出Bad Case用于重训练
"""

import os
import json
import csv
import gradio as gr
import numpy as np
from PIL import Image
from datetime import datetime

from detection_engine import (
    YOLODetector, VLMDetector, DetectionResult,
    CLASS_NAMES, CLASS_NAMES_CN, create_detector,
)
from db_manager import InspectionDB
from data_export import export_bad_cases, export_for_retraining

# ========== 全局配置 ==========
BASE_DIR = os.path.dirname(__file__)
# 模型查找顺序: models/best.pt (GPU训练导出) > runs/steel_train (本地训练) > 旧模型
_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "best.pt"),
    os.path.join(BASE_DIR, "runs", "steel_train", "weights", "best.pt"),
    os.path.join(BASE_DIR, "yolo-cases", "runs", "detect", "train31", "weights", "best.pt"),
]
DEFAULT_YOLO_MODEL = next((p for p in _MODEL_CANDIDATES if os.path.exists(p)),
                          _MODEL_CANDIDATES[0])
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "yolo-cases", "steel_data", "test", "images")

# 全局检测器实例 (延迟初始化)
yolo_detector = None
vlm_detector = None
db = InspectionDB()


def get_yolo_detector(model_path=None):
    """获取或创建YOLO检测器"""
    global yolo_detector
    path = model_path or DEFAULT_YOLO_MODEL
    if yolo_detector is None and os.path.exists(path):
        yolo_detector = YOLODetector(path)
    return yolo_detector


def get_vlm_detector(api_key=None, base_url=None, model_name=None):
    """获取或创建VLM检测器, 参数变化时自动重建"""
    global vlm_detector
    if vlm_detector is None:
        vlm_detector = VLMDetector(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )
    else:
        need_recreate = False
        if api_key and api_key != vlm_detector.api_key:
            need_recreate = True
        if base_url and base_url != vlm_detector.base_url:
            need_recreate = True
        if model_name and model_name != vlm_detector.model_name:
            need_recreate = True
        if need_recreate:
            vlm_detector = VLMDetector(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
            )
    return vlm_detector


# ========== Tab1: 缺陷检测 ==========
def run_detection(image, model_type, conf_threshold, vlm_api_key,
                  vlm_base_url, vlm_model_name):
    """执行检测"""
    if image is None:
        return None, "请上传图片", ""

    # 保存上传的图片到临时路径
    temp_dir = os.path.join(BASE_DIR, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_path = os.path.join(temp_dir, f"upload_{timestamp}.jpg")

    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(temp_path)
    elif isinstance(image, str):
        temp_path = image
    else:
        Image.fromarray(image).save(temp_path)

    try:
        if model_type == "YOLO":
            detector = get_yolo_detector()
            if detector is None:
                return None, f"YOLO模型文件不存在: {DEFAULT_YOLO_MODEL}", ""
            result = detector.detect(temp_path, conf=conf_threshold)
        else:
            detector = get_vlm_detector(
                api_key=vlm_api_key or None,
                base_url=vlm_base_url or None,
                model_name=vlm_model_name or None,
            )
            result = detector.detect(temp_path)

        # 保存到数据库
        log_id = db.save_detection(result)

        # 构建结果文本
        result_text = _format_result(result, log_id)

        annotated = result.annotated_image
        return annotated, result_text, str(log_id)

    except Exception as e:
        return None, f"检测失败: {str(e)}", ""


def _format_result(result, log_id):
    """格式化检测结果为文本"""
    lines = []
    lines.append(f"记录ID: {log_id}")
    lines.append(f"检测模型: {result.model_type.upper()}")
    lines.append(f"推理耗时: {result.inference_time:.3f} 秒")
    lines.append("-" * 40)

    if result.model_type == "yolo":
        if result.boxes:
            lines.append(f"检测到 {len(result.boxes)} 个缺陷:")
            for i, box in enumerate(result.boxes, 1):
                cn_name = CLASS_NAMES_CN.get(box.class_id, box.class_name)
                lines.append(
                    f"  [{i}] {cn_name} ({box.class_name})"
                    f"  置信度: {box.confidence:.2%}"
                    f"  位置: ({box.x1:.0f},{box.y1:.0f})-({box.x2:.0f},{box.y2:.0f})"
                )
        else:
            lines.append("未检测到缺陷")
    else:
        lines.append("VLM 分析结果:")
        lines.append(result.vlm_text)
        if result.boxes:
            lines.append("-" * 40)
            lines.append(f"定位到 {len(result.boxes)} 个缺陷区域:")
            for i, box in enumerate(result.boxes, 1):
                cn_name = CLASS_NAMES_CN.get(box.class_id, box.class_name)
                lines.append(
                    f"  [{i}] {cn_name} ({box.class_name})"
                    f"  位置: ({box.x1:.0f},{box.y1:.0f})-({box.x2:.0f},{box.y2:.0f})"
                )

    return "\n".join(lines)


def save_review(log_id_str, review_status, review_note):
    """保存审核结果"""
    if not log_id_str:
        return "请先执行检测"
    try:
        log_id = int(log_id_str)
        status_map = {
            "正确": "correct",
            "误检(检测错误)": "wrong",
            "漏检(未检出)": "missed",
        }
        status = status_map.get(review_status, "pending")
        db.update_review(log_id, status, review_note)
        return f"审核已保存 (ID={log_id}, 状态={review_status})"
    except Exception as e:
        return f"保存失败: {str(e)}"


def load_test_image(image_name):
    """从测试集加载图片"""
    if not image_name:
        return None
    img_path = os.path.join(TEST_IMAGES_DIR, image_name)
    if os.path.exists(img_path):
        return Image.open(img_path)
    return None


def get_test_image_list():
    """获取测试集图片列表"""
    if os.path.exists(TEST_IMAGES_DIR):
        files = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
        return files[:50]  # 限制显示前50张
    return []


# ========== Tab2: 检测记录 ==========
def load_records(model_filter, status_filter):
    """加载检测记录"""
    model = None if model_filter == "全部" else model_filter.lower()
    status_map = {
        "全部": "all", "待审核": "pending", "正确": "correct",
        "误检": "wrong", "漏检": "missed",
    }
    status = status_map.get(status_filter, "all")

    records = db.get_records(model_type=model, review_status=status, limit=50)

    if not records:
        return "暂无记录"

    lines = []
    lines.append(f"共 {len(records)} 条记录 (最多显示50条)\n")
    lines.append(f"{'ID':>5} | {'模型':>6} | {'缺陷数':>5} | {'耗时':>8} | {'状态':>6} | {'图片':>15} | {'时间'}")
    lines.append("-" * 85)

    status_cn = {
        "pending": "待审核", "correct": "正确",
        "wrong": "误检", "missed": "漏检",
    }

    for r in records:
        lines.append(
            f"{r['id']:>5} | {r['model_type']:>6} | {r['num_detections']:>5} | "
            f"{r['inference_time']:>7.3f}s | {status_cn.get(r['review_status'], r['review_status']):>6} | "
            f"{r['image_name']:>15} | {r['created_at']}"
        )

    return "\n".join(lines)


def review_record(record_id_str, review_status, review_note):
    """审核指定记录"""
    if not record_id_str:
        return "请输入记录ID"
    try:
        record_id = int(record_id_str)
        status_map = {
            "正确": "correct", "误检": "wrong", "漏检": "missed",
        }
        status = status_map.get(review_status, "pending")
        db.update_review(record_id, status, review_note)
        return f"审核完成: ID={record_id}, 状态={review_status}"
    except ValueError:
        return "请输入有效的记录ID(数字)"
    except Exception as e:
        return f"审核失败: {str(e)}"


def view_record_detail(record_id_str):
    """查看记录详情"""
    if not record_id_str:
        return "请输入记录ID", None
    try:
        record_id = int(record_id_str)
        record = db.get_record_by_id(record_id)
        if not record:
            return f"未找到ID={record_id}的记录", None

        lines = []
        lines.append(f"记录ID: {record['id']}")
        lines.append(f"图片: {record['image_name']}")
        lines.append(f"模型: {record['model_type']}")
        lines.append(f"检测数: {record['num_detections']}")
        lines.append(f"耗时: {record['inference_time']:.3f}s")
        lines.append(f"审核状态: {record['review_status']}")
        lines.append(f"审核备注: {record['review_note']}")
        lines.append(f"时间: {record['created_at']}")
        lines.append("-" * 40)

        if record['model_type'] == 'yolo':
            try:
                boxes = json.loads(record['detections_json'])
                for i, box in enumerate(boxes, 1):
                    lines.append(f"  [{i}] {box['class_name']} conf={box['confidence']:.2%}")
            except (json.JSONDecodeError, TypeError):
                pass
        else:
            lines.append(record.get('vlm_text', ''))

        # 尝试加载图片
        img = None
        img_path = record['image_path']
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path))

        return "\n".join(lines), img

    except ValueError:
        return "请输入有效的记录ID(数字)", None


# ========== Tab3: 统计分析 ==========
def generate_statistics():
    """生成统计报告"""
    stats = db.get_statistics()

    lines = []
    lines.append("=" * 50)
    lines.append("AI质检工作台 - 统计报告")
    lines.append("=" * 50)
    lines.append(f"\n检测总数: {stats['total']}")

    lines.append("\n按模型类型:")
    for model, count in stats.get('by_model', {}).items():
        lines.append(f"  {model}: {count} 次")

    lines.append("\n按审核状态:")
    status_cn = {"pending": "待审核", "correct": "正确", "wrong": "误检", "missed": "漏检"}
    for status, count in stats.get('by_status', {}).items():
        lines.append(f"  {status_cn.get(status, status)}: {count} 条")

    lines.append("\n平均推理时间:")
    for model, avg_time in stats.get('avg_inference_time', {}).items():
        lines.append(f"  {model}: {avg_time:.3f} 秒")

    if stats.get('accuracy') is not None:
        lines.append(f"\n审核准确率: {stats['accuracy']}% ({stats['reviewed_count']} 条已审核)")

    lines.append("\nYOLO检测缺陷类别分布:")
    for cls_name, count in sorted(stats.get('defect_class_counts', {}).items(),
                                   key=lambda x: -x[1]):
        cn = CLASS_NAMES_CN.get(
            {v: k for k, v in CLASS_NAMES.items()}.get(cls_name), cls_name
        )
        lines.append(f"  {cls_name} ({cn}): {count} 个")

    return "\n".join(lines)


# ========== Tab4: 数据导出 ==========
def do_export_bad_cases(model_filter):
    """导出Bad Case"""
    model = None if model_filter == "全部" else model_filter.lower()
    output_dir, count = export_bad_cases(model_type=model)
    if count > 0:
        return f"导出成功! 共 {count} 条Bad Case\n导出目录: {output_dir}"
    else:
        return "没有Bad Case需要导出 (请先在检测记录中标记误检或漏检)"


def do_export_retrain():
    """导出重训练数据"""
    output_dir, count = export_for_retraining()
    if count > 0:
        return f"导出成功! 共 {count} 张图片\n导出目录: {output_dir}"
    else:
        return "没有需要导出的数据 (请先在检测记录中标记误检或漏检)"


# ========== Tab5: 竞赛提交 ==========
def generate_submission(conf_threshold, iou_threshold, use_tta, progress=gr.Progress()):
    """
    对测试集进行批量预测, 生成 submission.csv

    submission.csv格式:
      image_id, bbox, category_id, confidence
      每行1个检测框, 同张图片的多个检测框分别记录在不同行
    """
    detector = get_yolo_detector()
    if detector is None:
        return "YOLO模型文件不存在, 请检查路径", None, ""

    if not os.path.exists(TEST_IMAGES_DIR):
        return f"测试集目录不存在: {TEST_IMAGES_DIR}", None, ""

    test_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR)
                         if f.endswith(('.jpg', '.png'))])
    if not test_files:
        return "测试集目录为空", None, ""

    output_path = os.path.join(BASE_DIR, "submission.csv")
    all_rows = []
    total_boxes = 0

    progress(0, desc="正在预测测试集...")

    for idx, img_file in enumerate(test_files):
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        image_id = int(os.path.splitext(img_file)[0])

        # 使用ultralytics原生predict, 支持TTA
        results = detector.model.predict(
            source=img_path,
            conf=conf_threshold,
            iou=iou_threshold,
            augment=use_tta,
            verbose=False,
        )

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for box, conf, cls_id in zip(xyxy, confs, classes):
                bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                all_rows.append({
                    "image_id": image_id,
                    "bbox": str(bbox),
                    "category_id": int(cls_id),
                    "confidence": float(conf),
                })
                total_boxes += 1

        progress((idx + 1) / len(test_files),
                 desc=f"已完成 {idx + 1}/{len(test_files)} 张")

    # 写入CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "bbox", "category_id", "confidence"])
        writer.writeheader()
        writer.writerows(all_rows)

    # 统计信息
    lines = []
    lines.append(f"预测完成!")
    lines.append(f"  测试图片: {len(test_files)} 张")
    lines.append(f"  检测到缺陷框: {total_boxes} 个")
    lines.append(f"  平均每张图: {total_boxes / max(len(test_files), 1):.1f} 个框")
    lines.append(f"  置信度阈值: {conf_threshold}")
    lines.append(f"  IoU阈值: {iou_threshold}")
    lines.append(f"  TTA增强: {'是' if use_tta else '否'}")
    lines.append(f"  输出文件: {output_path}")

    # 按类别统计
    class_counter = {}
    for row in all_rows:
        cid = row["category_id"]
        cname = CLASS_NAMES.get(cid, f"class_{cid}")
        class_counter[cname] = class_counter.get(cname, 0) + 1

    if class_counter:
        lines.append(f"\n各类别检测数量:")
        for cname, count in sorted(class_counter.items(), key=lambda x: -x[1]):
            cn = CLASS_NAMES_CN.get(
                {v: k for k, v in CLASS_NAMES.items()}.get(cname), cname
            )
            lines.append(f"  {cname} ({cn}): {count}")

    # 预览前10行
    lines.append(f"\nCSV预览 (前10行):")
    lines.append(f"{'image_id':>10} | {'bbox':>25} | {'category_id':>12} | {'confidence':>10}")
    lines.append("-" * 65)
    for row in all_rows[:10]:
        lines.append(
            f"{row['image_id']:>10} | {row['bbox']:>25} | "
            f"{row['category_id']:>12} | {row['confidence']:>10.4f}"
        )
    if len(all_rows) > 10:
        lines.append(f"  ... 共 {len(all_rows)} 行")

    summary = "\n".join(lines)
    return summary, output_path, output_path


# ========== 构建 Gradio 界面 ==========
def build_app():
    """构建Gradio界面"""

    with gr.Blocks(
        title="AI质检工作台",
        theme=gr.themes.Soft(),
        css="""
        .result-text { font-family: 'Consolas', 'Courier New', monospace; }
        """
    ) as app:

        gr.Markdown("# AI质检工作台 - 钢铁表面缺陷检测")
        gr.Markdown("支持 **YOLO目标检测** 和 **Qwen-VL多模态大模型** 两种检测引擎, 可对比分析")

        with gr.Tabs():

            # ===== Tab1: 缺陷检测 =====
            with gr.Tab("缺陷检测"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="上传钢铁表面图片",
                            type="numpy",
                            height=300,
                        )

                        # 模型选择
                        model_type = gr.Radio(
                            choices=["YOLO", "VLM (Qwen-VL)"],
                            value="YOLO",
                            label="选择检测引擎",
                        )

                        conf_slider = gr.Slider(
                            minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                            label="YOLO置信度阈值",
                            visible=True,
                        )

                        # VLM配置 (可折叠)
                        with gr.Accordion("VLM API配置", open=False):
                            vlm_api_key = gr.Textbox(
                                label="API Key",
                                value=os.environ.get("DASHSCOPE_API_KEY", ""),
                                placeholder="未设置DASHSCOPE_API_KEY环境变量, 请手动填入",
                                type="password",
                            )
                            vlm_base_url = gr.Textbox(
                                label="API Base URL",
                                value="https://dashscope.aliyuncs.com/compatible-mode/v1",
                            )
                            vlm_model_name = gr.Textbox(
                                label="模型名称",
                                value="qwen3-vl-plus",
                            )

                        detect_btn = gr.Button("开始检测", variant="primary", size="lg")

                        # 从测试集选择图片
                        with gr.Accordion("从测试集选择图片", open=False):
                            test_images = get_test_image_list()
                            test_dropdown = gr.Dropdown(
                                choices=test_images,
                                label="选择测试图片",
                                interactive=True,
                            )
                            load_test_btn = gr.Button("加载选中图片")

                    with gr.Column(scale=1):
                        output_image = gr.Image(label="检测结果", height=300)
                        result_text = gr.Textbox(
                            label="检测详情",
                            lines=10,
                            interactive=False,
                            elem_classes="result-text",
                        )

                        # 审核区域
                        gr.Markdown("### 人工审核")
                        current_log_id = gr.Textbox(visible=False)
                        with gr.Row():
                            review_status = gr.Radio(
                                choices=["正确", "误检(检测错误)", "漏检(未检出)"],
                                label="审核结果",
                            )
                        review_note = gr.Textbox(
                            label="审核备注",
                            placeholder="备注具体问题, 如: 左上角划痕未检出",
                        )
                        review_btn = gr.Button("提交审核")
                        review_result = gr.Textbox(label="审核状态", interactive=False)

                # 事件绑定
                detect_btn.click(
                    fn=run_detection,
                    inputs=[input_image, model_type, conf_slider,
                            vlm_api_key, vlm_base_url, vlm_model_name],
                    outputs=[output_image, result_text, current_log_id],
                )

                review_btn.click(
                    fn=save_review,
                    inputs=[current_log_id, review_status, review_note],
                    outputs=[review_result],
                )

                load_test_btn.click(
                    fn=load_test_image,
                    inputs=[test_dropdown],
                    outputs=[input_image],
                )

                def toggle_conf_slider(model):
                    return gr.update(visible=(model == "YOLO"))

                model_type.change(
                    fn=toggle_conf_slider,
                    inputs=[model_type],
                    outputs=[conf_slider],
                )

            # ===== Tab2: 检测记录 =====
            with gr.Tab("检测记录"):
                with gr.Row():
                    rec_model_filter = gr.Dropdown(
                        choices=["全部", "yolo", "vlm"],
                        value="全部",
                        label="模型筛选",
                    )
                    rec_status_filter = gr.Dropdown(
                        choices=["全部", "待审核", "正确", "误检", "漏检"],
                        value="全部",
                        label="审核状态筛选",
                    )
                    refresh_btn = gr.Button("刷新记录")

                records_display = gr.Textbox(
                    label="检测记录列表",
                    lines=15,
                    interactive=False,
                    elem_classes="result-text",
                )

                gr.Markdown("### 查看记录详情 / 审核")
                with gr.Row():
                    with gr.Column(scale=1):
                        detail_id = gr.Textbox(label="输入记录ID查看详情")
                        view_detail_btn = gr.Button("查看详情")
                        detail_text = gr.Textbox(
                            label="记录详情",
                            lines=12,
                            interactive=False,
                            elem_classes="result-text",
                        )
                    with gr.Column(scale=1):
                        detail_image = gr.Image(label="原始图片", height=250)

                with gr.Row():
                    batch_review_id = gr.Textbox(label="记录ID")
                    batch_review_status = gr.Dropdown(
                        choices=["正确", "误检", "漏检"],
                        label="审核结果",
                    )
                    batch_review_note = gr.Textbox(label="备注")
                    batch_review_btn = gr.Button("提交审核")
                batch_review_result = gr.Textbox(label="审核状态", interactive=False)

                refresh_btn.click(
                    fn=load_records,
                    inputs=[rec_model_filter, rec_status_filter],
                    outputs=[records_display],
                )

                view_detail_btn.click(
                    fn=view_record_detail,
                    inputs=[detail_id],
                    outputs=[detail_text, detail_image],
                )

                batch_review_btn.click(
                    fn=review_record,
                    inputs=[batch_review_id, batch_review_status, batch_review_note],
                    outputs=[batch_review_result],
                )

            # ===== Tab3: 统计分析 =====
            with gr.Tab("统计分析"):
                stats_btn = gr.Button("生成统计报告", variant="primary")
                stats_display = gr.Textbox(
                    label="统计报告",
                    lines=20,
                    interactive=False,
                    elem_classes="result-text",
                )

                stats_btn.click(
                    fn=generate_statistics,
                    outputs=[stats_display],
                )

            # ===== Tab4: 数据导出 =====
            with gr.Tab("数据导出"):
                gr.Markdown("""
                ### Bad Case 导出
                将审核标记为**误检**或**漏检**的图片导出, 用于后续重新标注和训练

                **工作流:**
                1. 在"缺陷检测"中检测图片
                2. 在"检测记录"中审核 (标记正确/误检/漏检)
                3. 在此处导出Bad Case
                4. 使用标注工具重新标注
                5. 加入训练集, 重新训练模型
                """)

                with gr.Row():
                    export_model_filter = gr.Dropdown(
                        choices=["全部", "yolo", "vlm"],
                        value="全部",
                        label="筛选模型类型",
                    )
                    export_bad_btn = gr.Button("导出Bad Case", variant="primary")

                export_retrain_btn = gr.Button("导出重训练数据包")
                export_result = gr.Textbox(label="导出结果", lines=5, interactive=False)

                export_bad_btn.click(
                    fn=do_export_bad_cases,
                    inputs=[export_model_filter],
                    outputs=[export_result],
                )

                export_retrain_btn.click(
                    fn=do_export_retrain,
                    outputs=[export_result],
                )

            # ===== Tab5: 竞赛提交 =====
            with gr.Tab("竞赛提交"):
                gr.Markdown("""
                ### 批量预测测试集 & 生成 submission.csv
                对测试集全部 **400张** 图片进行YOLO预测, 生成竞赛提交文件 `submission.csv`

                **提交格式:** 每行1个检测框 (image_id, bbox, category_id, confidence)
                """)

                with gr.Row():
                    sub_conf = gr.Slider(
                        minimum=0.01, maximum=0.9, value=0.25, step=0.01,
                        label="置信度阈值 (conf)",
                        info="降低阈值可提高召回率, 检出更多缺陷",
                    )
                    sub_iou = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.45, step=0.05,
                        label="NMS IoU阈值",
                        info="控制重叠框的合并程度",
                    )
                    sub_tta = gr.Checkbox(
                        label="启用TTA (测试时增强)",
                        value=False,
                        info="可提升精度, 但推理速度变慢约2-3倍",
                    )

                submit_btn = gr.Button(
                    "开始预测并生成 submission.csv",
                    variant="primary",
                    size="lg",
                )

                submit_summary = gr.Textbox(
                    label="预测结果摘要",
                    lines=20,
                    interactive=False,
                    elem_classes="result-text",
                )

                submit_file = gr.File(label="下载 submission.csv")
                submit_path_display = gr.Textbox(visible=False)

                submit_btn.click(
                    fn=generate_submission,
                    inputs=[sub_conf, sub_iou, sub_tta],
                    outputs=[submit_summary, submit_file, submit_path_display],
                )

    return app


# ========== 启动 ==========
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=7860,
        share=False,
        inbrowser=True,
    )
