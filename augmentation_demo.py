# -*- coding: utf-8 -*-
"""
数据增强可视化示例
展示各种数据增强技术对钢铁缺陷图片的效果, 帮助理解增强原理

用法:
    python augmentation_demo.py
    python augmentation_demo.py --image yolo-cases/steel_data/train/images/1.jpg
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "yolo-cases", "steel_data", "train", "images")
LABELS_DIR = os.path.join(BASE_DIR, "yolo-cases", "steel_data", "train", "labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "augmentation_output")

CLASS_NAMES = {
    0: "crazing(龟裂)", 1: "inclusion(夹杂)", 2: "pitted_surface(点蚀)",
    3: "scratches(划痕)", 4: "patches(斑块)", 5: "rolled-in_scale(氧化铁皮压入)",
}


# ========================================================================
# 基础增强函数 (纯 PIL 实现, 不依赖额外库)
# ========================================================================

def augment_horizontal_flip(img):
    """水平翻转: 缺陷可能出现在左右任意位置"""
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def augment_vertical_flip(img):
    """垂直翻转: 钢铁表面缺陷方向不固定"""
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def augment_rotate_90(img):
    """旋转90度: 增加方向多样性, 划痕可以是水平/垂直/斜向"""
    return img.rotate(90, expand=False)


def augment_rotate_random(img):
    """随机小角度旋转: 模拟相机轻微偏转"""
    angle = random.uniform(-20, 20)
    return img.rotate(angle, fillcolor=128, expand=False)


def augment_brightness(img):
    """亮度调整: 模拟光照变化, 工业相机光源不稳定"""
    factor = random.uniform(0.6, 1.4)
    return ImageEnhance.Brightness(img).enhance(factor)


def augment_contrast(img):
    """对比度调整: 增强缺陷可见性, 有些缺陷与背景对比度很低"""
    factor = random.uniform(0.6, 1.5)
    return ImageEnhance.Contrast(img).enhance(factor)


def augment_gaussian_noise(img):
    """高斯噪声: 模拟传感器噪声, 增加模型鲁棒性"""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 15, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def augment_blur(img):
    """模糊: 模拟相机失焦或运动模糊"""
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))


def augment_sharpen(img):
    """锐化: 增强缺陷边缘, 使裂纹/划痕更清晰"""
    return img.filter(ImageFilter.SHARPEN)


def augment_random_erasing(img):
    """随机擦除: 遮挡部分区域, 迫使模型学习更多特征而非记忆"""
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    # 随机擦除1-3个小区域
    for _ in range(random.randint(1, 3)):
        eh = random.randint(h // 8, h // 4)
        ew = random.randint(w // 8, w // 4)
        ey = random.randint(0, h - eh)
        ex = random.randint(0, w - ew)
        arr[ey:ey + eh, ex:ex + ew] = random.randint(0, 255)
    return Image.fromarray(arr)


def augment_cutout(img):
    """Cutout: 用黑色方块遮挡, 防止模型过度依赖局部特征"""
    draw = ImageDraw.Draw(img.copy())
    result = img.copy()
    draw = ImageDraw.Draw(result)
    w, h = img.size
    size = min(w, h) // 4
    cx = random.randint(size, w - size)
    cy = random.randint(size, h - size)
    draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=0)
    return result


def augment_hsv_shift(img):
    """HSV色彩空间变换: 模拟不同材料批次导致的色调差异"""
    arr = np.array(img)
    if len(arr.shape) == 2:
        arr = np.stack([arr] * 3, axis=-1)

    import colorsys
    h_shift = random.uniform(-0.02, 0.02)
    s_scale = random.uniform(0.5, 1.5)
    v_scale = random.uniform(0.7, 1.3)

    arr = arr.astype(np.float32) / 255.0
    # 简化的HSV变换: 直接在RGB空间近似调整
    arr[:, :, 0] = np.clip(arr[:, :, 0] * v_scale, 0, 1)
    if arr.shape[2] >= 2:
        arr[:, :, 1] = np.clip(arr[:, :, 1] * s_scale, 0, 1)
    if arr.shape[2] >= 3:
        arr[:, :, 2] = np.clip(arr[:, :, 2] * v_scale, 0, 1)

    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def augment_scale(img):
    """缩放: 模拟不同拍摄距离, 缺陷大小变化"""
    w, h = img.size
    scale = random.uniform(0.7, 1.3)
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = img.resize((new_w, new_h), Image.LANCZOS)

    # 中心裁剪或填充回原始尺寸
    result = Image.new(img.mode, (w, h), 128)
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2
    if scale < 1:
        result.paste(scaled, (paste_x, paste_y))
    else:
        crop_x = (new_w - w) // 2
        crop_y = (new_h - h) // 2
        result = scaled.crop((crop_x, crop_y, crop_x + w, crop_y + h))

    return result


# ========================================================================
# Mosaic 数据增强 (YOLO 的核心增强技术)
# ========================================================================

def augment_mosaic(images_dir, exclude_name=None):
    """
    Mosaic 数据增强: 将4张图拼成1张
    这是 YOLO 系列最重要的数据增强技术之一

    优点:
    1. 一张训练图中包含4种不同场景, 大幅增加数据多样性
    2. 增强 BatchNorm 的效果 (相当于变相增大 batch size)
    3. 小目标出现概率提高, 提升小目标检测能力
    """
    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    if exclude_name:
        all_images = [f for f in all_images if f != exclude_name]

    if len(all_images) < 4:
        return None

    selected = random.sample(all_images, 4)
    target_size = 200

    mosaic = Image.new('RGB', (target_size * 2, target_size * 2))

    for i, img_name in enumerate(selected):
        img = Image.open(os.path.join(images_dir, img_name)).convert('RGB')
        img = img.resize((target_size, target_size), Image.LANCZOS)
        row, col = divmod(i, 2)
        mosaic.paste(img, (col * target_size, row * target_size))

    return mosaic


# ========================================================================
# 可视化展示
# ========================================================================

def demo_single_augmentations(image_path, output_dir):
    """展示单张图片的各种增强效果"""
    img = Image.open(image_path).convert('RGB')
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # 读取标注信息
    label_path = os.path.join(LABELS_DIR, f"{basename}.txt")
    class_info = ""
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
            cls_ids = set()
            for line in lines:
                parts = line.strip().split()
                if parts:
                    cls_ids.add(int(parts[0]))
            class_info = ", ".join([CLASS_NAMES.get(c, str(c)) for c in cls_ids])

    augmentations = [
        ("原图", None, f"原始钢铁表面图片\n缺陷类型: {class_info}"),
        ("水平翻转", augment_horizontal_flip,
         "缺陷可能出现在左右任意位置\n水平翻转不改变缺陷特征"),
        ("垂直翻转", augment_vertical_flip,
         "钢铁缺陷方向不固定\n垂直翻转增加方向多样性"),
        ("旋转90度", augment_rotate_90,
         "划痕可以是水平/垂直/斜向\n旋转让模型学习各种方向"),
        ("随机旋转", augment_rotate_random,
         "模拟相机轻微偏转\n小角度旋转(-20度~+20度)"),
        ("亮度调整", augment_brightness,
         "模拟光照变化\n工业相机光源可能不稳定"),
        ("对比度调整", augment_contrast,
         "增强缺陷可见性\n有些缺陷与背景对比度很低"),
        ("高斯噪声", augment_gaussian_noise,
         "模拟传感器噪声\n增加模型鲁棒性"),
        ("模糊", augment_blur,
         "模拟相机失焦\n让模型不依赖清晰边缘"),
        ("锐化", augment_sharpen,
         "增强缺陷边缘\n使裂纹/划痕更清晰"),
        ("随机擦除", augment_random_erasing,
         "随机遮挡部分区域\n迫使模型学习更多特征"),
        ("缩放变换", augment_scale,
         "模拟不同拍摄距离\n缺陷在图中大小变化"),
    ]

    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    fig.suptitle(f"数据增强效果展示 - {basename}", fontsize=16, fontweight='bold')

    for idx, (name, aug_fn, desc) in enumerate(augmentations):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        if aug_fn is None:
            display_img = img
        else:
            display_img = aug_fn(img.copy())

        ax.imshow(display_img)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel(desc, fontsize=8, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"augmentation_demo_{basename}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 单图增强展示 -> {out_path}")


def demo_mosaic(images_dir, output_dir):
    """展示 Mosaic 数据增强效果"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Mosaic 数据增强 - YOLO 核心增强技术", fontsize=16, fontweight='bold')

    for i in range(3):
        mosaic = augment_mosaic(images_dir)
        if mosaic is not None:
            axes[i].imshow(mosaic)
            axes[i].set_title(f"Mosaic 样本 {i + 1}", fontsize=12)
            axes[i].set_xlabel("4张图拼成1张, 增加数据多样性\n提升小目标检测能力", fontsize=9, color='gray')
            axes[i].set_xticks([])
            axes[i].set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(output_dir, "mosaic_demo.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Mosaic增强展示 -> {out_path}")


def demo_augmentation_comparison(images_dir, output_dir):
    """对比: 同一张图经过多次随机增强, 每次结果都不同"""
    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    if not all_images:
        print("[!] 图片目录为空")
        return

    img_name = random.choice(all_images)
    img = Image.open(os.path.join(images_dir, img_name)).convert('RGB')
    basename = os.path.splitext(img_name)[0]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"随机增强组合 - 同一张图每次增强结果不同 ({img_name})",
                 fontsize=14, fontweight='bold')

    all_augments = [
        augment_horizontal_flip, augment_vertical_flip,
        augment_rotate_random, augment_brightness, augment_contrast,
        augment_gaussian_noise, augment_blur, augment_random_erasing,
        augment_scale,
    ]

    # 第一行: 原图 + 单次增强
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("原图", fontsize=11, fontweight='bold')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    for i in range(1, 5):
        aug_fn = random.choice(all_augments)
        augmented = aug_fn(img.copy())
        axes[0, i].imshow(augmented)
        axes[0, i].set_title(f"单次增强 {i}", fontsize=10)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

    # 第二行: 组合增强 (连续应用多个增强)
    for i in range(5):
        combined = img.copy()
        applied = []
        n_augs = random.randint(2, 4)
        selected_augs = random.sample(all_augments, n_augs)
        for aug_fn in selected_augs:
            combined = aug_fn(combined)
            applied.append(aug_fn.__name__.replace("augment_", ""))

        axes[1, i].imshow(combined)
        label = "+".join(applied[:2])
        if len(applied) > 2:
            label += f"+{len(applied) - 2}more"
        axes[1, i].set_title(f"组合增强", fontsize=10)
        axes[1, i].set_xlabel(label, fontsize=7, color='gray')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"augmentation_comparison_{basename}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 增强组合对比 -> {out_path}")


def demo_why_augmentation_matters(images_dir, labels_dir, output_dir):
    """
    展示: 为什么数据增强对钢铁缺陷检测特别重要

    钢铁缺陷数据集的三大挑战:
    1. 类内差异大: 同类缺陷外观差异很大 (如划痕可以水平/垂直/斜向)
    2. 类间相似: 不同类缺陷看起来相似 (如龟裂和点蚀)
    3. 数据量少: 每类只有200多张, 远不够深度学习模型的需求
    """
    from collections import defaultdict

    # 按类别收集样本
    class_samples = defaultdict(list)
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        basename = os.path.splitext(label_file)[0]
        img_path = os.path.join(images_dir, f"{basename}.jpg")
        if not os.path.exists(img_path):
            continue

        with open(os.path.join(labels_dir, label_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if len(class_samples[cls_id]) < 3:
                        class_samples[cls_id].append(img_path)

    # 展示类内差异
    n_classes = min(len(class_samples), 6)
    fig, axes = plt.subplots(n_classes, 3, figsize=(12, n_classes * 3))
    fig.suptitle("类内差异大 - 同类缺陷外观差异很大, 数据增强可以增加多样性",
                 fontsize=14, fontweight='bold')

    for row, cls_id in enumerate(sorted(class_samples.keys())[:n_classes]):
        for col in range(min(3, len(class_samples[cls_id]))):
            img = Image.open(class_samples[cls_id][col]).convert('RGB')
            ax = axes[row, col] if n_classes > 1 else axes[col]
            ax.imshow(img)
            if col == 0:
                ax.set_ylabel(CLASS_NAMES.get(cls_id, str(cls_id)),
                              fontsize=9, rotation=0, labelpad=80)
            ax.set_title(f"样本 {col + 1}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(output_dir, "why_augmentation_matters.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 增强必要性展示 -> {out_path}")


def demo_yolo_builtin_augmentations(output_dir):
    """
    说明 YOLO 内置的数据增强参数含义

    在 model.train() 中可以通过参数控制:
    """
    info = """
    ================================================================
    YOLO 训练时内置的数据增强参数 (Ultralytics)
    ================================================================

    在 model.train() 中通过参数控制, 训练时自动应用:

    [几何变换]
    fliplr=0.5       水平翻转概率 (缺陷左右位置不固定, 建议保持0.5)
    flipud=0.5       垂直翻转概率 (钢铁表面缺陷方向不固定, 建议0.5)
    degrees=15.0     随机旋转角度范围 (模拟相机偏转, 建议10-20度)
    translate=0.15   随机平移比例 (模拟目标位置偏移)
    scale=0.5        随机缩放比例 (模拟不同拍摄距离)
    shear=5.0        剪切变换角度 (轻微的透视变形)
    perspective=0.001 透视变换强度 (模拟非正对拍摄)

    [颜色变换]
    hsv_h=0.02       色调偏移 (模拟不同材料批次的色差)
    hsv_s=0.7        饱和度变化 (模拟光照变化)
    hsv_v=0.5        明度变化 (模拟亮度变化)

    [高级增强]
    mosaic=1.0       Mosaic增强概率 (4张图拼1张, YOLO的核心增强)
    mixup=0.15       MixUp概率 (两张图半透明叠加)
    copy_paste=0.15  Copy-Paste概率 (将目标复制粘贴到其他图)
    erasing=0.3      随机擦除概率 (防止过拟合)

    [训练策略]
    cos_lr=True      余弦退火学习率 (训练更稳定)
    warmup_epochs=3  预热轮数 (防止初始学习率过大)

    ================================================================
    对应代码:

    model.train(
        data="dataset.yaml",
        epochs=150, batch=16, imgsz=640,
        fliplr=0.5, flipud=0.5, degrees=15.0,
        mosaic=1.0, mixup=0.15, copy_paste=0.15,
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.5,
        erasing=0.3, cos_lr=True,
    )
    ================================================================
    """
    print(info)

    out_path = os.path.join(output_dir, "yolo_augmentation_params.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(info)
    print(f"[OK] YOLO增强参数说明 -> {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="数据增强可视化示例")
    parser.add_argument("--image", type=str, default=None,
                        help="指定图片路径, 默认自动从数据集选取")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 选择示例图片
    if args.image and os.path.exists(args.image):
        image_path = args.image
    else:
        all_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
        if not all_images:
            print("[!] 未找到图片, 请检查 yolo-cases/steel_data/train/images/ 目录")
            return
        image_path = os.path.join(IMAGES_DIR, random.choice(all_images))

    print("=" * 60)
    print("数据增强可视化示例")
    print("=" * 60)
    print(f"示例图片: {image_path}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    # 1. 单图增强效果展示 (12种增强)
    print("[1/5] 生成单图增强效果展示...")
    demo_single_augmentations(image_path, OUTPUT_DIR)

    # 2. Mosaic 增强展示
    print("[2/5] 生成 Mosaic 增强展示...")
    demo_mosaic(IMAGES_DIR, OUTPUT_DIR)

    # 3. 随机增强组合对比
    print("[3/5] 生成随机增强组合对比...")
    demo_augmentation_comparison(IMAGES_DIR, OUTPUT_DIR)

    # 4. 为什么数据增强重要
    print("[4/5] 生成增强必要性展示 (类内差异)...")
    demo_why_augmentation_matters(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR)

    # 5. YOLO内置增强参数说明
    print("[5/5] 生成 YOLO 内置增强参数说明...")
    demo_yolo_builtin_augmentations(OUTPUT_DIR)

    print()
    print("=" * 60)
    print("全部完成! 可视化结果已保存到:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
