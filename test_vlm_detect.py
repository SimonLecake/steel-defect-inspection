# -*- coding: utf-8 -*-
"""
VLM缺陷检测 独立测试脚本
使用 qwen3-vl-plus 模型对单张图片进行缺陷检测, 输出缺陷类型和位置, 并保存标注图。

用法:
    python test_vlm_detect.py
    python test_vlm_detect.py --image path/to/image.jpg
    python test_vlm_detect.py --image path/to/image.jpg --model qwen3-vl-plus --save output.jpg
"""

import argparse
import os
import sys

from detection_engine import VLMDetector, CLASS_NAMES_CN
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="VLM缺陷检测独立测试")
    parser.add_argument(
        "--image",
        default=os.path.join("yolo-cases", "steel_data", "train", "images", "0.jpg"),
        help="待检测图片路径 (默认: yolo-cases/steel_data/train/images/0.jpg)",
    )
    parser.add_argument("--model", default=None, help="模型名称 (默认: qwen3-vl-plus)")
    parser.add_argument("--save", default=None, help="标注图保存路径 (默认: vlm_result_<原文件名>)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[错误] 图片不存在: {args.image}")
        sys.exit(1)

    img = Image.open(args.image)
    print(f"图片: {args.image}")
    print(f"尺寸: {img.size[0]} x {img.size[1]}")
    print("-" * 50)

    detector = VLMDetector(model_name=args.model)
    print(f"模型: {detector.model_name}")
    print(f"API: {detector.base_url}")
    print("正在检测...")
    print("-" * 50)

    result = detector.detect(args.image)

    print(f"推理耗时: {result.inference_time:.3f} 秒")
    print(f"检测到缺陷数: {len(result.boxes)}")
    print("-" * 50)

    if result.boxes:
        for i, box in enumerate(result.boxes, 1):
            cn_name = CLASS_NAMES_CN.get(box.class_id, box.class_name)
            print(
                f"  [{i}] {cn_name} ({box.class_name})\n"
                f"      位置: ({box.x1:.0f}, {box.y1:.0f}) - ({box.x2:.0f}, {box.y2:.0f})\n"
                f"      宽高: {box.x2 - box.x1:.0f} x {box.y2 - box.y1:.0f} px"
            )
    else:
        print("  未检测到缺陷")

    print("-" * 50)
    print("VLM分析摘要:")
    print(result.vlm_text)

    # 保存标注图
    if result.annotated_image is not None:
        save_path = args.save
        if save_path is None:
            basename = os.path.splitext(os.path.basename(args.image))[0]
            save_path = f"vlm_result_{basename}.jpg"
        Image.fromarray(result.annotated_image).save(save_path)
        print(f"\n标注图已保存: {save_path}")


if __name__ == "__main__":
    main()
