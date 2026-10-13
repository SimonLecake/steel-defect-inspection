# -*- coding: utf-8 -*-
"""
检测引擎封装
统一接口封装 YOLO 和 Qwen-VL 两种检测方式

YOLO: 专用目标检测模型, 输出精确的边界框和类别
Qwen-VL: 多模态大模型, 输出自然语言描述和缺陷分析
"""

import os
import json
import re
import time
import base64
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ========================================================================
# YOLOv12 AAttn 兼容性补丁
# 不同版本的 ultralytics 中 AAttn 存在两种实现:
#   - pip 官方版: 使用 self.qkv (合并的 QKV)
#   - YOLOv12 源码版: 使用 self.qk + self.v (分离的 QK 和 V)
# 这里对 forward 方法做兼容, 使两种权重格式都能正常加载和推理
# ========================================================================
def _patch_aattn_forward():
    """修补 AAttn.forward, 兼容 qkv 和 qk+v 两种权重格式"""
    try:
        from ultralytics.nn.modules.block import AAttn
    except ImportError:
        return

    _original_forward = AAttn.forward

    def _compatible_forward(self, x):
        # 如果有 qkv 属性, 走原始逻辑
        if hasattr(self, 'qkv'):
            return _original_forward(self, x)

        # 兼容 qk + v 分离格式
        import torch
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x).flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape

        q, k = (
            qk.view(B, N, self.num_heads, self.head_dim * 2)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim], dim=2)
        )
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)

    AAttn.forward = _compatible_forward

_patch_aattn_forward()

# 缺陷类别定义
CLASS_NAMES = {
    0: "crazing", 1: "inclusion", 2: "pitted_surface",
    3: "scratches", 4: "patches", 5: "rolled-in_scale",
}
CLASS_NAMES_CN = {
    0: "龟裂", 1: "夹杂", 2: "点蚀",
    3: "划痕", 4: "斑块", 5: "氧化铁皮压入",
}

# 每个类别的显示颜色
CLASS_COLORS = {
    0: (255, 0, 0),       # 红
    1: (0, 255, 0),       # 绿
    2: (0, 0, 255),       # 蓝
    3: (255, 255, 0),     # 黄
    4: (255, 0, 255),     # 紫
    5: (0, 255, 255),     # 青
}

# 英文类别名 -> 类别ID 反向映射
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}


@dataclass
class DetectionBox:
    """单个检测框"""
    class_id: int
    class_name: str
    confidence: float
    # xyxy格式 (像素坐标)
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class DetectionResult:
    """检测结果"""
    model_type: str          # "yolo" 或 "vlm"
    image_path: str
    boxes: list = field(default_factory=list)  # List[DetectionBox]
    vlm_text: str = ""       # VLM的文本分析结果
    vlm_defect_types: list = field(default_factory=list)  # VLM识别的缺陷类型
    inference_time: float = 0.0
    annotated_image: Optional[np.ndarray] = None

    def to_dict(self):
        """转为可序列化的字典"""
        return {
            "model_type": self.model_type,
            "image_path": self.image_path,
            "boxes": [
                {
                    "class_id": b.class_id,
                    "class_name": b.class_name,
                    "confidence": round(b.confidence, 4),
                    "bbox": [round(b.x1, 1), round(b.y1, 1),
                             round(b.x2, 1), round(b.y2, 1)],
                }
                for b in self.boxes
            ],
            "vlm_text": self.vlm_text,
            "vlm_defect_types": self.vlm_defect_types,
            "inference_time": round(self.inference_time, 3),
        }


# ========================================================================
# YOLO 检测引擎
# ========================================================================
class YOLODetector:
    """YOLO目标检测器"""

    def __init__(self, model_path):
        from ultralytics import YOLO
        print(f"[YOLO] 加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.model_path = model_path

    def detect(self, image_path, conf=0.25, iou=0.45):
        """
        检测单张图片
        返回 DetectionResult
        """
        start_time = time.time()
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        elapsed = time.time() - start_time

        boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            det_boxes = results[0].boxes
            xyxy = det_boxes.xyxy.cpu().numpy()
            confs = det_boxes.conf.cpu().numpy()
            classes = det_boxes.cls.cpu().numpy()

            for box, c, cls_id in zip(xyxy, confs, classes):
                cls_id = int(cls_id)
                boxes.append(DetectionBox(
                    class_id=cls_id,
                    class_name=CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    confidence=float(c),
                    x1=float(box[0]), y1=float(box[1]),
                    x2=float(box[2]), y2=float(box[3]),
                ))

        result = DetectionResult(
            model_type="yolo",
            image_path=image_path,
            boxes=boxes,
            inference_time=elapsed,
        )

        result.annotated_image = self._draw_boxes(image_path, boxes)
        return result

    def _draw_boxes(self, image_path, boxes):
        """在图片上绘制检测框"""
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for box in boxes:
            color = CLASS_COLORS.get(box.class_id, (255, 255, 255))
            draw.rectangle([box.x1, box.y1, box.x2, box.y2],
                           outline=color, width=2)
            label = f"{CLASS_NAMES_CN.get(box.class_id, box.class_name)} {box.confidence:.2f}"
            draw.text((box.x1, max(0, box.y1 - 14)), label,
                      fill=color, font=font)

        return np.array(img)


# ========================================================================
# VLM 检测引擎 (Qwen-VL 通过 OpenAI 兼容 API)
# ========================================================================
class VLMDetector:
    """
    多模态大模型检测器
    通过 OpenAI 兼容 API 调用 Qwen-VL
    支持: DashScope、本地vLLM、任何OpenAI兼容接口
    """

    DEFAULT_PROMPT = """你是一个钢铁表面缺陷检测专家。请仔细分析这张钢铁表面图片,检测所有缺陷并用边界框标注位置。

可能的缺陷类型:
- crazing (龟裂): 表面呈现网状细小裂纹
- inclusion (夹杂): 表面有异物嵌入或杂质
- pitted_surface (点蚀): 表面有凹坑或麻点
- scratches (划痕): 表面有线状刮痕
- patches (斑块): 表面有不规则色斑或区域变色
- rolled-in_scale (氧化铁皮压入): 表面有氧化皮被压入的痕迹

请检测图中所有缺陷并返回其位置坐标,输出格式如下:
[{"bbox_2d": [x1, y1, x2, y2], "label": "缺陷英文名", "description": "简要描述缺陷特征"}]

如果没有检测到缺陷,返回空数组: []"""

    def __init__(self, api_key=None, base_url=None, model_name=None):
        """
        初始化VLM检测器

        参数:
            api_key: API密钥
            base_url: API地址
                - DashScope: https://dashscope.aliyuncs.com/compatible-mode/v1
                - 本地vLLM: http://localhost:8000/v1
            model_name: 模型名称
                - DashScope: qwen-vl-max / qwen2.5-vl-72b-instruct
                - 本地: 取决于部署的模型
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_name = model_name or os.environ.get(
            "VLM_MODEL_NAME", "qwen3-vl-plus"
        )
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError:
            print("[!] 需要安装: pip install openai")
        except Exception as e:
            print(f"[!] VLM客户端初始化失败: {e}")

    def _encode_image(self, image_path):
        """将图片编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _extract_json(self, text):
        """从VLM回复中提取JSON字符串, 去除thinking标签和markdown代码块"""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts[1::2]:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                return cleaned
        return text.strip()

    def _convert_bbox(self, bbox_2d, img_width, img_height):
        """
        将VLM返回的坐标转换为像素坐标
        Qwen3-VL: 0-1000 归一化坐标, 需要乘以图片尺寸
        Qwen2.5-VL: 直接返回绝对像素坐标
        """
        if "qwen3" in self.model_name.lower():
            x1 = bbox_2d[0] / 1000 * img_width
            y1 = bbox_2d[1] / 1000 * img_height
            x2 = bbox_2d[2] / 1000 * img_width
            y2 = bbox_2d[3] / 1000 * img_height
        else:
            x1, y1, x2, y2 = bbox_2d
        return x1, y1, x2, y2

    def _parse_bbox_response(self, reply, img_width, img_height):
        """
        解析VLM返回的bbox_2d格式响应
        返回: (boxes, defect_types, summary_text)
        """
        json_str = self._extract_json(reply)
        parsed = json.loads(json_str)

        if isinstance(parsed, dict):
            parsed = [parsed]

        boxes = []
        defect_types = []
        desc_lines = []

        for item in parsed:
            bbox_2d = item.get("bbox_2d")
            label = item.get("label", "unknown")
            description = item.get("description", "")

            if not bbox_2d or len(bbox_2d) != 4:
                continue

            x1, y1, x2, y2 = self._convert_bbox(bbox_2d, img_width, img_height)
            class_id = CLASS_NAME_TO_ID.get(label, -1)

            boxes.append(DetectionBox(
                class_id=class_id,
                class_name=label,
                confidence=1.0,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

            if label not in defect_types:
                defect_types.append(label)

            cn_name = CLASS_NAMES_CN.get(class_id, label)
            desc_lines.append(f"- {cn_name} ({label}): {description}")

        has_defect = len(boxes) > 0
        summary = f"缺陷检测: {'有缺陷' if has_defect else '无缺陷'}\n"
        summary += f"检测到 {len(boxes)} 个缺陷区域\n"
        if desc_lines:
            summary += "\n".join(desc_lines)

        return boxes, defect_types, summary

    def detect(self, image_path, prompt=None):
        """使用VLM分析图片, 返回带有边界框的 DetectionResult"""
        if self.client is None:
            return DetectionResult(
                model_type="vlm",
                image_path=image_path,
                vlm_text="[错误] VLM客户端未初始化, 请检查API配置",
            )

        prompt = prompt or self.DEFAULT_PROMPT
        b64_image = self._encode_image(image_path)

        img = Image.open(image_path)
        img_width, img_height = img.size

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            elapsed = time.time() - start_time
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            elapsed = time.time() - start_time
            return DetectionResult(
                model_type="vlm",
                image_path=image_path,
                vlm_text=f"[API调用失败] {str(e)}",
                inference_time=elapsed,
            )

        # 解析VLM返回的bbox_2d响应
        boxes = []
        defect_types = []
        description = reply
        try:
            boxes, defect_types, description = self._parse_bbox_response(
                reply, img_width, img_height
            )
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            description = f"VLM原始回复 (JSON解析失败):\n{reply}"

        result = DetectionResult(
            model_type="vlm",
            image_path=image_path,
            boxes=boxes,
            vlm_text=description,
            vlm_defect_types=defect_types,
            inference_time=elapsed,
        )

        result.annotated_image = self._draw_boxes(image_path, boxes)
        return result

    def _draw_boxes(self, image_path, boxes):
        """在图片上绘制VLM检测到的缺陷框"""
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for box in boxes:
            color = CLASS_COLORS.get(box.class_id, (255, 255, 255))
            draw.rectangle([box.x1, box.y1, box.x2, box.y2],
                           outline=color, width=2)
            cn_name = CLASS_NAMES_CN.get(box.class_id, box.class_name)
            label = f"{cn_name} ({box.class_name})"
            draw.text((box.x1, max(0, box.y1 - 16)), label,
                      fill=color, font=font)

        return np.array(img)


# ========================================================================
# 工厂函数
# ========================================================================
def create_detector(model_type, **kwargs):
    """
    创建检测器的工厂函数

    参数:
        model_type: "yolo" 或 "vlm"
        kwargs: 传递给对应检测器的参数
    """
    if model_type == "yolo":
        model_path = kwargs.get("model_path")
        if not model_path:
            raise ValueError("YOLO检测器需要指定model_path")
        return YOLODetector(model_path)
    elif model_type == "vlm":
        return VLMDetector(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
            model_name=kwargs.get("model_name"),
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
