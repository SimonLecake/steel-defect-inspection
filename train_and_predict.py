# -*- coding: utf-8 -*-
"""
YOLOv12 钢铁缺陷检测 - 训练 + 预测 + 生成 submission.csv
自动检测 CPU/GPU, 一个脚本搞定全流程

========== 使用方法 ==========

1. 本地 CPU 跑通流程 (1个epoch, 验证代码能跑):
   python train_and_predict.py

2. AutoDL GPU 训练 (正式训练):
   python train_and_predict.py --device 0 --epochs 100 --batch 16

3. 只做预测 (用已训练好的模型生成 submission.csv):
   python train_and_predict.py --predict_only --model ./runs/steel_train/weights/best.pt

4. 增强训练 (更多数据增强, 推荐竞赛使用):
   python train_and_predict.py --device 0 --epochs 150 --batch 16 --enhanced

5. 预测时启用TTA (测试时增强, 可提分):
   python train_and_predict.py --predict_only --model ./runs/steel_train/weights/best.pt --tta

========== AutoDL GPU 训练完整流程 ==========
Step1: 上传项目文件夹到 /root/autodl-tmp/
Step2: cd /root/autodl-tmp/26-项目实战：AI质检
Step3: pip install ultralytics
Step4: python train_and_predict.py --device 0 --epochs 100 --batch 16
Step5: 训练完成后, 模型保存在 runs/steel_train/weights/best.pt
Step6: 下载 best.pt 和 submission.csv 到本地
Step7: 将 best.pt 放到本地 models/ 目录下
Step8: 本地运行: python train_and_predict.py --predict_only --model ./models/best.pt
"""

import os
import csv
import time
import yaml
import shutil
import argparse
import tempfile
from pathlib import Path

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STEEL_DATA_DIR = os.path.join(BASE_DIR, "yolo-cases", "steel_data")
IMAGES_DIR = os.path.join(STEEL_DATA_DIR, "train", "images")
LABELS_DIR = os.path.join(STEEL_DATA_DIR, "train", "labels")
TEST_IMAGES_DIR = os.path.join(STEEL_DATA_DIR, "test", "images")
YOLOV12_YAML = os.path.join(BASE_DIR, "yolo-cases", "yolov12.yaml")
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASS_NAMES = {
    0: "crazing", 1: "inclusion", 2: "pitted_surface",
    3: "scratches", 4: "patches", 5: "rolled-in_scale",
}


def create_dataset_yaml():
    """
    动态生成 dataset.yaml, 自动适配当前机器的路径
    解决 autodl 和本地路径不一致的问题
    """
    yaml_content = {
        "path": STEEL_DATA_DIR,
        "train": "train/train.txt",
        "val": "train/val.txt",
        "test": "train/test.txt",
        "names": CLASS_NAMES,
    }

    yaml_path = os.path.join(BASE_DIR, "dataset_local.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"[OK] 数据集配置已生成: {yaml_path}")
    print(f"     数据集路径: {STEEL_DATA_DIR}")
    return yaml_path


def detect_device(device_arg):
    """自动检测设备"""
    import torch

    if device_arg is not None:
        device = str(device_arg)
    elif torch.cuda.is_available():
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"[OK] 检测到GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    else:
        device = "cpu"
        print("[!] 未检测到GPU, 将使用CPU训练 (速度较慢)")

    return device


def train(args):
    """训练 YOLOv12 模型"""
    from ultralytics import YOLO

    device = detect_device(args.device)
    dataset_yaml = create_dataset_yaml()

    # 根据设备调整默认参数
    is_cpu = (device == "cpu")
    epochs = args.epochs if args.epochs else (1 if is_cpu else 100)
    batch = args.batch if args.batch else (4 if is_cpu else 16)
    workers = 0 if is_cpu else 8

    print()
    print("=" * 60)
    print("YOLOv12 钢铁缺陷检测 - 开始训练")
    print("=" * 60)
    print(f"  设备: {device}")
    print(f"  模型: yolov12n (从 yolov12.yaml 配置创建)")
    print(f"  轮数: {epochs}")
    print(f"  批次: {batch}")
    print(f"  输入尺寸: {args.imgsz}")
    print(f"  增强模式: {'增强版' if args.enhanced else '基线版'}")
    if is_cpu:
        print(f"  [CPU模式] 仅训练{epochs}轮验证流程, 正式训练请使用GPU")
    print()

    # 从yaml配置创建yolov12n模型
    # 注意: 不直接加载 yolov12n.pt, 因为不同ultralytics版本间
    # AAttn模块存在兼容性问题(qk vs qkv)
    model = YOLO(YOLOV12_YAML)

    # 训练参数
    train_kwargs = dict(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=args.imgsz,
        patience=args.patience if args.patience else (50 if not is_cpu else epochs),
        device=device,
        workers=workers,
        pretrained=False,
        save=True,
        verbose=True,
        plots=False,
        project=os.path.join(BASE_DIR, "runs"),
        name="steel_train",
        exist_ok=True,

        # 基础增强
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        scale=0.5,
        fliplr=0.5,
    )

    # 增强模式: 更多数据增强
    if args.enhanced:
        train_kwargs.update(
            flipud=0.5,
            degrees=15.0,
            translate=0.15,
            hsv_h=0.02,
            hsv_s=0.7,
            hsv_v=0.5,
            perspective=0.001,
            shear=5.0,
            erasing=0.3,
            cos_lr=True,
        )

    results = model.train(**train_kwargs)

    # 打印模型保存路径
    best_pt = os.path.join(BASE_DIR, "runs", "steel_train", "weights", "best.pt")
    print()
    print("=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"  最佳模型: {best_pt}")
    print()

    if not is_cpu:
        print("下一步:")
        print(f"  1. 模型已保存在: {best_pt}")
        print(f"  2. 可直接生成submission: python train_and_predict.py --predict_only --model \"{best_pt}\"")
        print(f"  3. 或将 best.pt 下载到本地 models/ 目录, 供 app.py 使用")

    return best_pt


def predict(model_path, args):
    """用训练好的模型预测测试集, 生成 submission.csv"""
    from ultralytics import YOLO

    if not os.path.exists(model_path):
        print(f"[!] 模型文件不存在: {model_path}")
        return

    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"[!] 测试集目录不存在: {TEST_IMAGES_DIR}")
        return

    print()
    print("=" * 60)
    print("批量预测测试集, 生成 submission.csv")
    print("=" * 60)

    model = YOLO(model_path)

    test_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR)
                         if f.endswith(('.jpg', '.png'))])
    print(f"  模型: {model_path}")
    print(f"  测试图片: {len(test_files)} 张")
    print(f"  conf={args.conf}, iou={args.iou}, TTA={args.tta}")
    print()

    all_rows = []
    start = time.time()

    for idx, img_file in enumerate(test_files):
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        image_id = int(os.path.splitext(img_file)[0])

        results = model.predict(
            source=img_path,
            conf=args.conf,
            iou=args.iou,
            augment=args.tta,
            verbose=False,
        )

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for box, conf_val, cls_id in zip(xyxy, confs, classes):
                bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                all_rows.append({
                    "image_id": image_id,
                    "bbox": str(bbox),
                    "category_id": int(cls_id),
                    "confidence": round(float(conf_val), 6),
                })

        if (idx + 1) % 50 == 0 or idx == len(test_files) - 1:
            elapsed = time.time() - start
            print(f"  进度: {idx + 1}/{len(test_files)}  "
                  f"检测框: {len(all_rows)}  耗时: {elapsed:.1f}s")

    # 写入 submission.csv
    output_path = os.path.join(BASE_DIR, "submission.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "bbox", "category_id", "confidence"])
        writer.writeheader()
        writer.writerows(all_rows)

    elapsed = time.time() - start

    # 统计
    print()
    print("=" * 60)
    print("预测完成!")
    print("=" * 60)
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  检测框总数: {len(all_rows)}")
    print(f"  平均每张图: {len(all_rows) / max(len(test_files), 1):.1f} 个框")
    print(f"  输出文件: {output_path}")

    # 按类别统计
    class_counter = {}
    for row in all_rows:
        cid = row["category_id"]
        cname = CLASS_NAMES.get(cid, f"class_{cid}")
        class_counter[cname] = class_counter.get(cname, 0) + 1

    if class_counter:
        print("\n  各类别检测数量:")
        for cname, count in sorted(class_counter.items(), key=lambda x: -x[1]):
            print(f"    {cname}: {count}")

    print(f"\n  --> 提交文件: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv12 钢铁缺陷检测 - 训练 & 预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  CPU 快速验证:    python train_and_predict.py
  GPU 正式训练:    python train_and_predict.py --device 0 --epochs 100 --batch 16
  增强训练:        python train_and_predict.py --device 0 --epochs 150 --batch 16 --enhanced
  仅预测:          python train_and_predict.py --predict_only --model ./runs/steel_train/weights/best.pt
  预测+TTA:        python train_and_predict.py --predict_only --model ./models/best.pt --tta
  调整置信度:      python train_and_predict.py --predict_only --model ./models/best.pt --conf 0.15
        """,
    )

    # 训练参数
    parser.add_argument("--device", type=str, default=None,
                        help="训练设备: 0(GPU) 或 cpu, 默认自动检测")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数, 默认: CPU=5, GPU=100")
    parser.add_argument("--batch", type=int, default=None,
                        help="批次大小, 默认: CPU=4, GPU=16")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸 (默认640)")
    parser.add_argument("--patience", type=int, default=None,
                        help="早停轮数")
    parser.add_argument("--enhanced", action="store_true",
                        help="启用增强数据增强 (推荐竞赛使用)")

    # 预测参数
    parser.add_argument("--predict_only", action="store_true",
                        help="仅预测, 不训练")
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径 (predict_only模式必填)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值 (默认0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU阈值 (默认0.45)")
    parser.add_argument("--tta", action="store_true",
                        help="启用测试时增强 (TTA)")
    parser.add_argument("--skip_predict", action="store_true",
                        help="训练后跳过预测")

    args = parser.parse_args()

    if args.predict_only:
        # 仅预测模式
        model_path = args.model
        if not model_path:
            # 自动寻找模型
            candidates = [
                os.path.join(BASE_DIR, "runs", "steel_train", "weights", "best.pt"),
                os.path.join(MODELS_DIR, "best.pt"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    model_path = p
                    break
            if not model_path:
                print("[!] 未找到模型, 请通过 --model 指定模型路径")
                return
        predict(model_path, args)
    else:
        # 训练 + 预测
        best_pt = train(args)

        if not args.skip_predict:
            predict(best_pt, args)


if __name__ == "__main__":
    main()
