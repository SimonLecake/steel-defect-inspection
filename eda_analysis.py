# -*- coding: utf-8 -*-
"""
钢铁表面缺陷数据集 EDA 分析
功能：类别分布统计、样本可视化、标注框分析、类别不均衡评估
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 配置 ==========
STEEL_DATA_DIR = os.path.join(os.path.dirname(__file__), "yolo-cases", "steel_data")
LABELS_DIR = os.path.join(STEEL_DATA_DIR, "train", "labels")
IMAGES_DIR = os.path.join(STEEL_DATA_DIR, "train", "images")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")

CLASS_NAMES = {
    0: "crazing (龟裂)",
    1: "inclusion (夹杂)",
    2: "pitted_surface (点蚀)",
    3: "scratches (划痕)",
    4: "patches (斑块)",
    5: "rolled-in_scale (氧化铁皮压入)",
}
CLASS_NAMES_SHORT = {
    0: "crazing", 1: "inclusion", 2: "pitted_surface",
    3: "scratches", 4: "patches", 5: "rolled-in_scale",
}


def load_all_labels(labels_dir):
    """读取所有标注文件, 返回 {filename: [(class_id, x, y, w, h), ...]}"""
    annotations = {}
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        basename = os.path.splitext(os.path.basename(label_file))[0]
        boxes = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append((cls_id, x, y, w, h))
        annotations[basename] = boxes
    return annotations


def analyze_class_distribution(annotations):
    """分析类别分布"""
    class_counts = Counter()
    boxes_per_image = []

    for basename, boxes in annotations.items():
        boxes_per_image.append(len(boxes))
        for cls_id, *_ in boxes:
            class_counts[cls_id] += 1

    print("=" * 60)
    print("类别分布统计")
    print("=" * 60)
    total = sum(class_counts.values())
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        ratio = count / total * 100
        bar = "#" * int(ratio)
        print(f"  {CLASS_NAMES[cls_id]:30s} | {count:5d} ({ratio:5.1f}%) {bar}")
    print(f"  {'总计':30s} | {total:5d}")
    print(f"\n图片数量: {len(annotations)}")
    print(f"每张图的标注框数: 最少{min(boxes_per_image)}, 最多{max(boxes_per_image)}, "
          f"平均{np.mean(boxes_per_image):.1f}")
    print()

    return class_counts, boxes_per_image


def analyze_box_sizes(annotations):
    """分析标注框尺寸分布"""
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    class_areas = defaultdict(list)

    for basename, boxes in annotations.items():
        for cls_id, x, y, w, h in boxes:
            widths.append(w)
            heights.append(h)
            area = w * h
            areas.append(area)
            aspect_ratios.append(w / max(h, 1e-6))
            class_areas[cls_id].append(area)

    print("=" * 60)
    print("标注框尺寸分析 (归一化坐标)")
    print("=" * 60)
    print(f"  宽度: 最小{min(widths):.3f}, 最大{max(widths):.3f}, 平均{np.mean(widths):.3f}")
    print(f"  高度: 最小{min(heights):.3f}, 最大{max(heights):.3f}, 平均{np.mean(heights):.3f}")
    print(f"  面积: 最小{min(areas):.4f}, 最大{max(areas):.4f}, 平均{np.mean(areas):.4f}")
    print()

    return widths, heights, areas, aspect_ratios, class_areas


def plot_class_distribution(class_counts, output_dir):
    """绘制类别分布柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    classes = sorted(class_counts.keys())
    names = [CLASS_NAMES_SHORT[c] for c in classes]
    counts = [class_counts[c] for c in classes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    bars = axes[0].bar(names, counts, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title("各类别缺陷数量", fontsize=14)
    axes[0].set_ylabel("标注框数量")
    axes[0].tick_params(axis='x', rotation=30)
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(count), ha='center', va='bottom', fontsize=10)

    axes[1].pie(counts, labels=names, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 9})
    axes[1].set_title("类别占比", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 类别分布图 -> {output_dir}/class_distribution.png")


def plot_box_analysis(widths, heights, areas, aspect_ratios, output_dir):
    """绘制标注框尺寸分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(widths, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title("标注框宽度分布")
    axes[0, 0].set_xlabel("宽度 (归一化)")

    axes[0, 1].hist(heights, bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title("标注框高度分布")
    axes[0, 1].set_xlabel("高度 (归一化)")

    axes[1, 0].hist(areas, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title("标注框面积分布")
    axes[1, 0].set_xlabel("面积 (归一化)")

    axes[1, 1].scatter(widths, heights, alpha=0.3, s=10, c='purple')
    axes[1, 1].set_title("宽度 vs 高度")
    axes[1, 1].set_xlabel("宽度")
    axes[1, 1].set_ylabel("高度")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "box_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 标注框分析图 -> {output_dir}/box_analysis.png")


def plot_sample_images(annotations, images_dir, output_dir, samples_per_class=2):
    """每类缺陷展示样本图片"""
    class_samples = defaultdict(list)
    for basename, boxes in annotations.items():
        for cls_id, *_ in boxes:
            if len(class_samples[cls_id]) < samples_per_class:
                class_samples[cls_id].append(basename)

    n_classes = len(class_samples)
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(4 * samples_per_class, 3 * n_classes))

    for row, cls_id in enumerate(sorted(class_samples.keys())):
        for col, basename in enumerate(class_samples[cls_id]):
            img_path = os.path.join(images_dir, f"{basename}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax = axes[row, col] if n_classes > 1 else axes[col]
                ax.imshow(img, cmap='gray')

                # 绘制标注框
                for c, x, y, w, h in annotations[basename]:
                    if c == cls_id:
                        img_w, img_h = img.size
                        x1 = (x - w / 2) * img_w
                        y1 = (y - h / 2) * img_h
                        rect_w = w * img_w
                        rect_h = h * img_h
                        rect = plt.Rectangle((x1, y1), rect_w, rect_h,
                                             linewidth=2, edgecolor='red', facecolor='none')
                        ax.add_patch(rect)

                ax.set_title(CLASS_NAMES_SHORT[cls_id], fontsize=10)
                ax.axis('off')

    plt.suptitle("各类缺陷样本展示", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_images.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 样本展示图 -> {output_dir}/sample_images.png")


def plot_class_area_distribution(class_areas, output_dir):
    """各类别缺陷面积箱线图"""
    fig, ax = plt.subplots(figsize=(10, 5))

    classes = sorted(class_areas.keys())
    data = [class_areas[c] for c in classes]
    names = [CLASS_NAMES_SHORT[c] for c in classes]

    bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_title("各类别缺陷面积分布", fontsize=14)
    ax.set_ylabel("面积 (归一化)")
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_area_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 类别面积分布图 -> {output_dir}/class_area_distribution.png")


def check_class_imbalance(class_counts):
    """评估类别不均衡程度"""
    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    ratio = max_count / max(min_count, 1)

    print("=" * 60)
    print("类别不均衡评估")
    print("=" * 60)
    print(f"  最多类别数量: {max_count}")
    print(f"  最少类别数量: {min_count}")
    print(f"  不均衡比例: {ratio:.1f}:1")
    if ratio > 5:
        print("  [!] 类别严重不均衡, 建议采取以下措施:")
        print("      1. 对少数类别进行过采样/数据增强")
        print("      2. 使用类别权重(class weights)")
        print("      3. 使用Focal Loss")
    elif ratio > 2:
        print("  [!] 存在一定的类别不均衡, 建议关注少数类别的召回率")
    else:
        print("  [OK] 类别分布较均衡")
    print()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("加载标注数据...")
    annotations = load_all_labels(LABELS_DIR)

    class_counts, boxes_per_image = analyze_class_distribution(annotations)
    widths, heights, areas, aspect_ratios, class_areas = analyze_box_sizes(annotations)
    check_class_imbalance(class_counts)

    print("生成可视化图表...")
    plot_class_distribution(class_counts, OUTPUT_DIR)
    plot_box_analysis(widths, heights, areas, aspect_ratios, OUTPUT_DIR)
    plot_sample_images(annotations, IMAGES_DIR, OUTPUT_DIR)
    plot_class_area_distribution(class_areas, OUTPUT_DIR)

    print(f"\n所有EDA分析结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
