# -*- coding: utf-8 -*-
"""
增强版 YOLO 训练脚本 - 钢铁缺陷检测
包含: 基线训练、增强训练、竞赛提分技巧、模型评估对比

实际工作中YOLO训练不只是 model.train() 一行代码，
这个脚本展示了从基线到竞赛级别的完整优化路径。
"""

import os
import shutil
import random
import argparse
import numpy as np
from pathlib import Path

# ========== 配置 ==========
BASE_DIR = os.path.dirname(__file__)
STEEL_DATA_DIR = os.path.join(BASE_DIR, "yolo-cases", "steel_data")
DATASET_YAML = os.path.join(STEEL_DATA_DIR, "dataset.yaml")


# ========================================================================
# 策略1: 基线训练 (大多数人停在这一步)
# ========================================================================
def train_baseline():
    """
    基线训练: 默认参数, 这是最简单的方式
    很多学员问"不就是调个API吗？" -- 对，但这只是起点
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("策略1: 基线训练 (默认参数)")
    print("=" * 60)

    model = YOLO("yolo12n.pt")  # 使用预训练的nano模型
    results = model.train(
        data=DATASET_YAML,
        epochs=100,
        batch=16,
        imgsz=640,
        patience=50,
        device="0",
        project=os.path.join(BASE_DIR, "runs_enhanced"),
        name="baseline",
    )
    return results


# ========================================================================
# 策略2: 数据增强调优 (核心提分手段)
# ========================================================================
def train_enhanced_augmentation():
    """
    增强数据增强参数
    为什么重要：钢铁缺陷的特点是纹理多变、方向不固定、大小差异大
    需要针对性地调整增强策略
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("策略2: 增强数据增强")
    print("=" * 60)
    print("调优思路:")
    print("  1. 钢铁缺陷可能出现在任何方向 -> 增加旋转和翻转")
    print("  2. 缺陷大小差异大 -> 增加多尺度训练")
    print("  3. 光照不均匀 -> 增加HSV增强")
    print("  4. 数据量有限(1400张) -> 加强Mosaic和MixUp")
    print()

    model = YOLO("yolo12n.pt")
    results = model.train(
        data=DATASET_YAML,
        epochs=150,
        batch=16,
        imgsz=640,
        patience=80,
        device="0",
        project=os.path.join(BASE_DIR, "runs_enhanced"),
        name="enhanced_aug",

        # --- 关键增强参数 ---
        # Mosaic: 4张图拼成1张, 增加小目标检测能力
        mosaic=1.0,
        # MixUp: 两张图混合, 增加模型泛化能力
        mixup=0.15,
        # Copy-Paste: 将目标复制粘贴到其他图片, 适合目标较少的场景
        copy_paste=0.15,
        # 缩放: 模拟不同距离拍摄
        scale=0.5,
        # 水平翻转: 缺陷可能出现在任何位置
        fliplr=0.5,
        # 垂直翻转: 钢铁表面缺陷方向不固定
        flipud=0.5,
        # 旋转: 缺陷角度多变
        degrees=15.0,
        # 平移: 模拟相机位置变化
        translate=0.15,
        # HSV色调增强
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        # 透视变换: 轻微的视角变化
        perspective=0.001,
        # 剪切变换
        shear=5.0,
        # 擦除: 随机遮挡部分区域, 增加鲁棒性
        erasing=0.3,
    )
    return results


# ========================================================================
# 策略3: 更大的模型 + 更大的输入尺寸
# ========================================================================
def train_larger_model():
    """
    使用更大的模型和输入尺寸
    权衡: 精度更高, 但速度更慢、显存需求更大
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("策略3: 更大的模型 (yolo12s) + 更大的输入 (1280)")
    print("=" * 60)
    print("注意: 需要更多显存, 建议>=12G GPU")
    print()

    model = YOLO("yolo12s.pt")  # small模型, 比nano更强
    results = model.train(
        data=DATASET_YAML,
        epochs=150,
        batch=8,        # 大图需要减小batch
        imgsz=1280,     # 更大的输入尺寸, 保留更多细节
        patience=80,
        device="0",
        project=os.path.join(BASE_DIR, "runs_enhanced"),
        name="larger_model",

        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        degrees=15.0,

        # 余弦退火学习率, 训练更稳定
        cos_lr=True,
    )
    return results


# ========================================================================
# 策略4: 离线数据增强 (扩充训练集)
# ========================================================================
def offline_augmentation(images_dir, labels_dir, output_dir, multiply=3):
    """
    离线数据增强: 在训练前对数据集进行扩充
    特别适合样本少的类别

    和YOLO内置增强的区别:
    - YOLO内置增强: 每个epoch随机变换, 不改变数据集大小
    - 离线增强: 物理扩充数据集, 增加数据量
    - 两者可以结合使用
    """
    try:
        import cv2
        import albumentations as A
    except ImportError:
        print("[!] 需要安装: pip install albumentations opencv-python")
        return

    print("=" * 60)
    print(f"策略4: 离线数据增强 (每张图扩充{multiply}倍)")
    print("=" * 60)

    out_images = os.path.join(output_dir, "images")
    out_labels = os.path.join(output_dir, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # 针对钢铁表面缺陷设计的增强管道
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # CLAHE: 对比度受限自适应直方图均衡化, 增强缺陷可见性
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30,
                           border_mode=cv2.BORDER_REFLECT, p=0.5),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,
    ))

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    total_generated = 0

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:5]])

        # 复制原图
        basename = os.path.splitext(img_file)[0]
        shutil.copy2(img_path, os.path.join(out_images, img_file))
        shutil.copy2(label_path, os.path.join(out_labels, basename + ".txt"))

        # 生成增强副本
        for i in range(multiply):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']

                if len(aug_bboxes) == 0:
                    continue

                aug_name = f"{basename}_aug{i}"
                aug_img = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(out_images, f"{aug_name}.jpg"), aug_img)

                with open(os.path.join(out_labels, f"{aug_name}.txt"), "w") as f:
                    for cls_id, bbox in zip(aug_labels, aug_bboxes):
                        f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                total_generated += 1
            except Exception as e:
                continue

    print(f"原始图片: {len(image_files)} 张")
    print(f"新增图片: {total_generated} 张")
    print(f"输出目录: {output_dir}")


# ========================================================================
# 策略5: 测试时增强 (TTA) - 推理阶段提分
# ========================================================================
def predict_with_tta(model_path, test_images_dir):
    """
    测试时增强 (Test Time Augmentation)
    原理: 对同一张图做多次不同变换, 取所有预测的并集
    优点: 不需要重新训练, 直接提升推理精度
    缺点: 推理速度变慢(约2-3倍)
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("策略5: 测试时增强 (TTA)")
    print("=" * 60)

    model = YOLO(model_path)

    # 普通预测
    print("普通预测:")
    results_normal = model.predict(
        source=test_images_dir,
        conf=0.25,
        iou=0.45,
        augment=False,
        save=True,
        project=os.path.join(BASE_DIR, "runs_enhanced"),
        name="predict_normal",
    )

    # TTA预测
    print("\nTTA预测:")
    results_tta = model.predict(
        source=test_images_dir,
        conf=0.25,
        iou=0.45,
        augment=True,  # 开启TTA
        save=True,
        project=os.path.join(BASE_DIR, "runs_enhanced"),
        name="predict_tta",
    )

    return results_normal, results_tta


# ========================================================================
# 策略6: 置信度阈值调优
# ========================================================================
def optimize_confidence_threshold(model_path):
    """
    置信度阈值调优
    默认conf=0.25, 但最优值取决于具体数据集
    - 降低阈值: 召回率提高, 但精度降低(适合"宁可多检不可漏检"的场景)
    - 提高阈值: 精度提高, 但召回率降低(适合减少误报的场景)
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("策略6: 置信度阈值调优")
    print("=" * 60)

    model = YOLO(model_path)

    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    print(f"{'阈值':>8s} | {'mAP50':>8s} | {'mAP50-95':>10s} | {'Precision':>10s} | {'Recall':>8s}")
    print("-" * 60)

    best_map50 = 0
    best_conf = 0.25

    for conf in thresholds:
        results = model.val(data=DATASET_YAML, conf=conf, verbose=False)
        map50 = results.box.map50
        map5095 = results.box.map
        precision = results.box.mp
        recall = results.box.mr

        marker = ""
        if map50 > best_map50:
            best_map50 = map50
            best_conf = conf
            marker = " <-- best"

        print(f"{conf:>8.2f} | {map50:>8.4f} | {map5095:>10.4f} | {precision:>10.4f} | {recall:>8.4f}{marker}")

    print(f"\n最优置信度阈值: {best_conf} (mAP50={best_map50:.4f})")
    return best_conf


# ========================================================================
# 竞赛技巧总结
# ========================================================================
def print_competition_tips():
    """打印竞赛提分技巧清单"""
    print()
    print("=" * 70)
    print("YOLO 缺陷检测竞赛提分技巧清单")
    print("=" * 70)
    tips = [
        ("数据层面", [
            "1. 数据EDA: 了解类别分布、缺陷大小、标注质量",
            "2. 类别不均衡处理: 过采样少数类 / 使用类别权重",
            "3. 离线数据增强: 用albumentations扩充数据集(特别是少数类)",
            "4. 数据清洗: 检查标注错误, 去除低质量样本",
            "5. 交叉验证: 使用K-Fold避免过拟合验证集",
        ]),
        ("训练层面", [
            "1. 预训练权重: 始终从COCO预训练权重开始(不要从零训练)",
            "2. 输入尺寸: 640->1280, 保留更多细节(需要更多显存)",
            "3. 增强参数调优: 根据数据特点调整(旋转/翻转/HSV等)",
            "4. 学习率: 使用余弦退火(cos_lr=True)",
            "5. 训练轮数: 适当增加epochs, 配合早停(patience)",
            "6. 多模型训练: 用不同尺寸的模型(n/s/m)分别训练",
        ]),
        ("推理层面", [
            "1. TTA(测试时增强): augment=True, 不需要重新训练",
            "2. 置信度调优: 不同conf阈值对结果影响很大",
            "3. NMS IoU调优: 调整iou阈值控制重叠框的合并",
            "4. 模型集成: 多个模型的预测结果进行WBF融合",
        ]),
        ("工业场景vs竞赛", [
            "竞赛追求: mAP分数最高",
            "工业追求: 漏检率最低 + 速度满足产线节拍",
            "工业场景通常会降低conf阈值, 宁可多检不可漏检",
            "工业还需要考虑: 模型大小、推理速度、边缘部署",
        ]),
    ]

    for category, items in tips:
        print(f"\n[{category}]")
        for item in items:
            print(f"  {item}")
    print()


def main():
    parser = argparse.ArgumentParser(description="增强版YOLO训练")
    parser.add_argument("--strategy", type=str, default="tips",
                        choices=["baseline", "enhanced", "larger", "augment",
                                 "tta", "conf_tune", "tips"],
                        help="训练策略")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join(BASE_DIR, "yolo-cases", "runs", "detect",
                                             "train31", "weights", "best.pt"),
                        help="已训练模型路径(用于TTA和阈值调优)")
    args = parser.parse_args()

    if args.strategy == "baseline":
        train_baseline()
    elif args.strategy == "enhanced":
        train_enhanced_augmentation()
    elif args.strategy == "larger":
        train_larger_model()
    elif args.strategy == "augment":
        output_dir = os.path.join(STEEL_DATA_DIR, "train_augmented")
        images_dir = os.path.join(STEEL_DATA_DIR, "train", "images")
        labels_dir = os.path.join(STEEL_DATA_DIR, "train", "labels")
        offline_augmentation(images_dir, labels_dir, output_dir, multiply=3)
    elif args.strategy == "tta":
        test_dir = os.path.join(STEEL_DATA_DIR, "test", "images")
        predict_with_tta(args.model_path, test_dir)
    elif args.strategy == "conf_tune":
        optimize_confidence_threshold(args.model_path)
    elif args.strategy == "tips":
        print_competition_tips()


if __name__ == "__main__":
    main()
