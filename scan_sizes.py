from pathlib import Path

def dir_size_mb(p):
    p = Path(p)
    if not p.exists():
        return None
    if p.is_file():
        return p.stat().st_size / 1024 / 1024
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1024 / 1024

targets = [
    ("dataset/final_split_yolo",              r"d:/2026物理竞赛/Physics/dataset/final_split_yolo"),
    ("dataset/final_split_yolo_sim_only",      r"d:/2026物理竞赛/Physics/dataset/final_split_yolo_sim_only"),
    ("dataset/final_split_yolo_sim1000_diff",  r"d:/2026物理竞赛/Physics/dataset/final_split_yolo_sim1000_diff"),
    ("dataset/real_data",                      r"d:/2026物理竞赛/Physics/dataset/real_data"),
    ("dataset/images/Auto_Generated",          r"d:/2026物理竞赛/Physics/dataset/images/Auto_Generated"),
    ("dataset/images/Final_Dataset",           r"d:/2026物理竞赛/Physics/dataset/images/Final_Dataset"),
    ("dataset/images/Final_Dataset_sv1",       r"d:/2026物理竞赛/Physics/dataset/images/Final_Dataset_smallscale_v1"),
    ("dataset/images/Final_Dataset_sv2",       r"d:/2026物理竞赛/Physics/dataset/images/Final_Dataset_smallscale_v2"),
    ("dataset/images/Real_Data",               r"d:/2026物理竞赛/Physics/dataset/images/Real_Data"),
    ("dataset/images/Defect platform results", r"d:/2026物理竞赛/Physics/dataset/images/Defect platform results"),
    ("runs/train/split (最早)",                r"d:/2026物理竞赛/Physics/dl_main_model_yolo/runs/train/yolo11s_final_split"),
    ("runs/train/split6",                      r"d:/2026物理竞赛/Physics/dl_main_model_yolo/runs/train/yolo11s_final_split6"),
    ("runs/train/split7",                      r"d:/2026物理竞赛/Physics/dl_main_model_yolo/runs/train/yolo11s_final_split7"),
    ("runs/train/split8",                      r"d:/2026物理竞赛/Physics/dl_main_model_yolo/runs/train/yolo11s_final_split8"),
    ("runs/train/sim1000_diff_fast",           r"d:/2026物理竞赛/Physics/dl_main_model_yolo/runs/train/yolo11s_sim1000_diff_fast"),
    ("runs/detect (根目录)",                   r"d:/2026物理竞赛/Physics/runs/detect"),
    ("runs/reports",                           r"d:/2026物理竞赛/Physics/dl_main_model_yolo/runs/reports"),
    ("OpticalDefectExperiment/",               r"d:/2026物理竞赛/Physics/OpticalDefectExperiment"),
    ("yolo11s.pt (根目录)",                    r"d:/2026物理竞赛/Physics/yolo11s.pt"),
    ("yolo26n.pt (根目录)",                    r"d:/2026物理竞赛/Physics/yolo26n.pt"),
    ("yolo26n.pt (dl目录)",                    r"d:/2026物理竞赛/Physics/dl_main_model_yolo/yolo26n.pt"),
    ("牛顿环瑕疵(6).zip",                      r"d:/2026物理竞赛/Physics/牛顿环瑕疵 (6).zip"),
    ("牛顿环瑕疵0513back.zip",                 r"d:/2026物理竞赛/Physics/牛顿环瑕疵0513back.zip"),
    ("build.spec",                             r"d:/2026物理竞赛/Physics/build.spec"),
    ("defect_labels.py",                       r"d:/2026物理竞赛/Physics/defect_labels.py"),
    ("__pycache__ (根目录)",                   r"d:/2026物理竞赛/Physics/__pycache__"),
    ("resources/",                             r"d:/2026物理竞赛/Physics/resources"),
]

for name, path in targets:
    mb = dir_size_mb(path)
    if mb is None:
        print(f"  N/A      {name}")
    elif mb >= 1:
        print(f"{mb:7.1f}MB  {name}")
    else:
        print(f"{mb*1024:7.1f}KB  {name}")
