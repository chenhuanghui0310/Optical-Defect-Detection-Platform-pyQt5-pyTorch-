# -*- coding: utf-8 -*-
"""
模型训练与测试结果报告生成脚本
生成包含评价指标说明的完整评估报告，方便添加到实验报告中。

用法：
    python generate_report.py --weights runs/train/yolo11s_final_split2/weights/best.pt
                               --data ../dataset/final_split_yolo/dataset.yaml
                               --split test
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# 确保可以导入 ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics 未安装，请运行: pip install ultralytics")
    sys.exit(1)


# ─────────────────────────────────────────────
# 评价指标说明（中文）
# ─────────────────────────────────────────────
METRIC_EXPLANATIONS = {
    "Precision（精确率）": (
        "精确率 = TP / (TP + FP)。\n"
        "在所有被模型预测为某类瑕疵的样本中，真正属于该类的比例。\n"
        "精确率高意味着误报少——模型不会把正常区域误判为瑕疵。"
    ),
    "Recall（召回率）": (
        "召回率 = TP / (TP + FN)。\n"
        "在所有真实存在某类瑕疵的样本中，被模型正确检测到的比例。\n"
        "召回率高意味着漏报少——模型能找到大多数真实瑕疵。"
    ),
    "mAP@0.5（平均精度均值，IoU=0.5）": (
        "mAP@0.5 是各类别 AP（Average Precision）在 IoU 阈值=0.5 时的均值。\n"
        "AP 是精确率-召回率曲线下的面积，综合衡量检测框的位置准确性和分类准确性。\n"
        "IoU=0.5 表示预测框与真实框的交并比 ≥ 0.5 才算正确检测。\n"
        "mAP@0.5 是目标检测领域最常用的综合评价指标，越接近1越好。"
    ),
    "mAP@0.5:0.95（严格平均精度均值）": (
        "mAP@0.5:0.95 是在 IoU 阈值从 0.5 到 0.95（步长0.05）下各 AP 的均值。\n"
        "比 mAP@0.5 更严格，要求检测框位置更精准。\n"
        "该指标是 COCO 数据集的标准评价指标，对定位精度要求更高。"
    ),
    "F1-Score（F1分数）": (
        "F1 = 2 × Precision × Recall / (Precision + Recall)。\n"
        "精确率和召回率的调和平均数，综合衡量两者的平衡性。\n"
        "当精确率和召回率都高时，F1 才高，适合评估不平衡数据集上的性能。"
    ),
    "Confusion Matrix（混淆矩阵）": (
        "混淆矩阵展示每个真实类别被预测为各类别的数量分布。\n"
        "对角线元素为正确分类数，非对角线元素为误分类数。\n"
        "通过混淆矩阵可以直观看出哪两类瑕疵容易被混淆。"
    ),
    "IoU（交并比）": (
        "IoU = 预测框与真实框的交集面积 / 两者的并集面积。\n"
        "用于衡量检测框的定位精度，范围 [0, 1]，越接近1定位越准确。\n"
        "IoU > 0.5 通常被认为是合格的检测结果。"
    ),
}

CLASS_NAMES_CN = {
    "scratch": "表面划痕",
    "bump":    "小凸起",
    "dent":    "小凹陷",
    "dust":    "灰尘污染",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成模型评估报告")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(
            Path(__file__).resolve().parent
            / "runs" / "train" / "yolo11s_final_split2" / "weights" / "best.pt"
        ),
        help="训练好的权重路径",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "dataset" / "final_split_yolo" / "dataset.yaml"
        ),
        help="数据集 yaml 路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="评估集合（默认 test）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA 设备 id 或 cpu",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "reports"),
        help="报告输出目录",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推理图像尺寸",
    )
    return parser.parse_args()


def run_validation(weights: str, data: str, split: str, device: str, imgsz: int):
    """运行 YOLO 验证并返回指标字典"""
    model = YOLO(weights)
    try:
        metrics = model.val(
            data=data,
            imgsz=imgsz,
            device=device,
            split=split,
            verbose=False,
            plots=True,
        )
    except Exception:
        # GPU 失败时回退 CPU
        metrics = model.val(
            data=data,
            imgsz=imgsz,
            device="cpu",
            split=split,
            verbose=False,
            plots=True,
        )
    return metrics


def extract_metrics(metrics) -> dict:
    """从 ultralytics metrics 对象中提取关键指标"""
    result = {}

    # 整体指标
    try:
        result["precision_mean"] = float(metrics.box.mp)
        result["recall_mean"] = float(metrics.box.mr)
        result["map50"] = float(metrics.box.map50)
        result["map50_95"] = float(metrics.box.map)
    except Exception:
        result["precision_mean"] = 0.0
        result["recall_mean"] = 0.0
        result["map50"] = 0.0
        result["map50_95"] = 0.0

    # F1
    p = result["precision_mean"]
    r = result["recall_mean"]
    result["f1_mean"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    # 各类别指标
    try:
        names = metrics.names  # {0: 'scratch', 1: 'bump', ...}
        ap50_per_class = metrics.box.ap50.tolist() if hasattr(metrics.box, "ap50") else []
        ap_per_class = metrics.box.ap.tolist() if hasattr(metrics.box, "ap") else []
        p_per_class = metrics.box.p.tolist() if hasattr(metrics.box, "p") else []
        r_per_class = metrics.box.r.tolist() if hasattr(metrics.box, "r") else []

        per_class = {}
        for i, name in names.items():
            cn = CLASS_NAMES_CN.get(name, name)
            pc = p_per_class[i] if i < len(p_per_class) else 0.0
            rc = r_per_class[i] if i < len(r_per_class) else 0.0
            f1c = (2 * pc * rc / (pc + rc)) if (pc + rc) > 0 else 0.0
            per_class[cn] = {
                "precision": pc,
                "recall": rc,
                "f1": f1c,
                "ap50": ap50_per_class[i] if i < len(ap50_per_class) else 0.0,
                "ap50_95": ap_per_class[i] if i < len(ap_per_class) else 0.0,
            }
        result["per_class"] = per_class
    except Exception as e:
        result["per_class"] = {}
        print(f"  [WARN] 无法提取各类别指标: {e}")

    return result


def format_report(
    metrics_dict: dict,
    weights_path: str,
    data_path: str,
    split: str,
    save_dir: str,
) -> str:
    """生成格式化的中文报告文本"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    lines.append("=" * 70)
    lines.append("        光学干涉瑕疵检测模型 — 评估报告")
    lines.append("=" * 70)
    lines.append(f"生成时间  : {now}")
    lines.append(f"模型权重  : {weights_path}")
    lines.append(f"数据集    : {data_path}")
    lines.append(f"评估集合  : {split}")
    lines.append(f"可视化结果: {save_dir}")
    lines.append("")

    # ── 一、评价指标说明 ──────────────────────────────────────────
    lines.append("─" * 70)
    lines.append("一、评价指标说明")
    lines.append("─" * 70)
    for metric_name, explanation in METRIC_EXPLANATIONS.items():
        lines.append(f"\n【{metric_name}】")
        for line in explanation.strip().split("\n"):
            lines.append(f"  {line}")
    lines.append("")

    # ── 二、整体性能指标 ─────────────────────────────────────────
    lines.append("─" * 70)
    lines.append("二、整体性能指标（所有类别均值）")
    lines.append("─" * 70)
    p = metrics_dict.get("precision_mean", 0)
    r = metrics_dict.get("recall_mean", 0)
    f1 = metrics_dict.get("f1_mean", 0)
    map50 = metrics_dict.get("map50", 0)
    map5095 = metrics_dict.get("map50_95", 0)

    lines.append(f"  Precision（精确率）    : {p:.4f}  ({p*100:.2f}%)")
    lines.append(f"  Recall（召回率）       : {r:.4f}  ({r*100:.2f}%)")
    lines.append(f"  F1-Score               : {f1:.4f}  ({f1*100:.2f}%)")
    lines.append(f"  mAP@0.5                : {map50:.4f}  ({map50*100:.2f}%)")
    lines.append(f"  mAP@0.5:0.95           : {map5095:.4f}  ({map5095*100:.2f}%)")
    lines.append("")

    # 性能评级
    if map50 >= 0.90:
        grade = "优秀 ★★★★★"
    elif map50 >= 0.80:
        grade = "良好 ★★★★"
    elif map50 >= 0.70:
        grade = "中等 ★★★"
    elif map50 >= 0.60:
        grade = "较差 ★★"
    else:
        grade = "需改进 ★"
    lines.append(f"  综合评级（基于mAP@0.5）: {grade}")
    lines.append("")

    # ── 三、各类别详细指标 ───────────────────────────────────────
    lines.append("─" * 70)
    lines.append("三、各类别详细指标")
    lines.append("─" * 70)
    per_class = metrics_dict.get("per_class", {})
    if per_class:
        header = f"  {'类别':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP@0.5':>10} {'AP@0.5:0.95':>12}"
        lines.append(header)
        lines.append("  " + "-" * 66)
        for cls_name, cls_metrics in per_class.items():
            row = (
                f"  {cls_name:<12}"
                f" {cls_metrics['precision']:>10.4f}"
                f" {cls_metrics['recall']:>10.4f}"
                f" {cls_metrics['f1']:>10.4f}"
                f" {cls_metrics['ap50']:>10.4f}"
                f" {cls_metrics['ap50_95']:>12.4f}"
            )
            lines.append(row)
    else:
        lines.append("  （无法获取各类别指标）")
    lines.append("")

    # ── 四、结果解读 ─────────────────────────────────────────────
    lines.append("─" * 70)
    lines.append("四、结果解读与建议")
    lines.append("─" * 70)

    # 自动生成解读
    if p > r + 0.1:
        lines.append("  • 精确率明显高于召回率：模型倾向于保守预测，漏报较多。")
        lines.append("    建议：降低置信度阈值（conf），或增加训练数据中的小尺寸样本。")
    elif r > p + 0.1:
        lines.append("  • 召回率明显高于精确率：模型倾向于激进预测，误报较多。")
        lines.append("    建议：提高置信度阈值（conf），或增加负样本训练数据。")
    else:
        lines.append("  • 精确率与召回率较为均衡，模型整体表现稳定。")

    if per_class:
        worst_cls = min(per_class.items(), key=lambda x: x[1]["ap50"])
        best_cls = max(per_class.items(), key=lambda x: x[1]["ap50"])
        lines.append(f"  • 表现最好的类别：{best_cls[0]}（AP@0.5 = {best_cls[1]['ap50']:.4f}）")
        lines.append(f"  • 表现最差的类别：{worst_cls[0]}（AP@0.5 = {worst_cls[1]['ap50']:.4f}）")
        if worst_cls[1]["ap50"] < 0.5:
            lines.append(f"    建议：针对 {worst_cls[0]} 类别增加训练样本，特别是0.01μm小尺寸样本。")

    lines.append("")
    lines.append("  • 混淆矩阵和PR曲线图已保存至可视化结果目录，可直接插入实验报告。")
    lines.append("")

    # ── 五、JSON 数据摘要 ────────────────────────────────────────
    lines.append("─" * 70)
    lines.append("五、机器可读数据（JSON格式，可直接复制到实验报告附录）")
    lines.append("─" * 70)
    json_data = {
        "report_time": now,
        "weights": weights_path,
        "split": split,
        "overall": {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map5095, 4),
        },
        "per_class": {
            k: {kk: round(vv, 4) for kk, vv in v.items()}
            for k, v in per_class.items()
        },
    }
    lines.append(json.dumps(json_data, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("=" * 70)
    lines.append("报告结束")
    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] 权重文件不存在: {weights_path}")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] 数据集 yaml 不存在: {data_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  光学干涉瑕疵检测模型评估报告生成器")
    print(f"{'='*60}")
    print(f"  权重: {weights_path}")
    print(f"  数据: {data_path}")
    print(f"  集合: {args.split}")
    print(f"  设备: {args.device}")
    print()

    print("[1/3] 运行模型验证...")
    metrics = run_validation(
        str(weights_path), str(data_path), args.split, args.device, args.imgsz
    )

    # 获取 ultralytics 保存的可视化结果目录
    try:
        save_dir = str(metrics.save_dir)
    except Exception:
        save_dir = str(output_dir)

    print("[2/3] 提取评价指标...")
    metrics_dict = extract_metrics(metrics)

    print("[3/3] 生成报告...")
    report_text = format_report(
        metrics_dict,
        str(weights_path),
        str(data_path),
        args.split,
        save_dir,
    )

    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = output_dir / f"evaluation_report_{args.split}_{timestamp}.txt"
    report_filename.write_text(report_text, encoding="utf-8")

    # 同时保存 JSON
    json_filename = output_dir / f"evaluation_metrics_{args.split}_{timestamp}.json"
    json_data = {
        "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weights": str(weights_path),
        "data": str(data_path),
        "split": args.split,
        "overall": {
            "precision": round(metrics_dict.get("precision_mean", 0), 4),
            "recall": round(metrics_dict.get("recall_mean", 0), 4),
            "f1": round(metrics_dict.get("f1_mean", 0), 4),
            "mAP50": round(metrics_dict.get("map50", 0), 4),
            "mAP50_95": round(metrics_dict.get("map50_95", 0), 4),
        },
        "per_class": {
            k: {kk: round(vv, 4) for kk, vv in v.items()}
            for k, v in metrics_dict.get("per_class", {}).items()
        },
    }
    json_filename.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 打印报告到控制台
    print()
    print(report_text)

    print(f"\n[OK] 报告已保存:")
    print(f"  文本报告: {report_filename}")
    print(f"  JSON数据: {json_filename}")
    print(f"  可视化图: {save_dir}")


if __name__ == "__main__":
    main()
