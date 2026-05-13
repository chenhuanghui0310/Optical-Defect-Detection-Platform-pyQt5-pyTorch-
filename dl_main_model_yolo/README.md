# Main YOLO model workspace

This folder isolates the primary defect-detection model workflow for competition use.

## Files

- `defect_competition.yaml`: dataset mapping to `../dataset`
- `train_main_model.py`: training entry
- `validate_main_model.py`: evaluation entry
- `predict_main_model.py`: inference entry
- `requirements.txt`: Python dependencies
- `runs/`: output root for train and prediction logs

## Quick start

1) Install dependencies:

```bash
cd d:/2026物理竞赛/Physics
.venv/Scripts/activate
pip install -r dl_main_model_yolo/requirements.txt
```

2) Train the main model:

```bash
python dl_main_model_yolo/train_main_model.py --model yolo11s.pt --epochs 120 --batch 16 --device 0
```

3) Validate best checkpoint:

```bash
python dl_main_model_yolo/validate_main_model.py --weights dl_main_model_yolo/runs/train/yolo11s_defect_main/weights/best.pt --device 0
```

4) Run prediction:

```bash
python dl_main_model_yolo/predict_main_model.py --weights dl_main_model_yolo/runs/train/yolo11s_defect_main/weights/best.pt --source dataset/images/val --device 0
```

## Use Final_Dataset with auto split

1) Auto split Final_Dataset and generate weak YOLO labels:

```bash
python dl_main_model_yolo/split_final_dataset.py --val-ratio 0.2 --seed 42 --clean
```

Output will be written to:

- `dataset/final_split_yolo/images/train`
- `dataset/final_split_yolo/images/val`
- `dataset/final_split_yolo/labels/train`
- `dataset/final_split_yolo/labels/val`
- `dataset/final_split_yolo/dataset.yaml`

2) Train on the split dataset:

```bash
python dl_main_model_yolo/train_final_split.py --model yolo11s.pt --epochs 120 --batch 16 --device 0
```

Note: The script currently builds weak labels (full-image boxes) from `defect_type` in JSON files. This is suitable for fast baseline experiments.

## Recommended tuning order

1. `--epochs` and `--batch`
2. `--imgsz`
3. model scale: `yolo11n.pt -> yolo11s.pt -> yolo11m.pt`
4. confidence threshold for inference (`--conf`)

## Notes

- Use your synthetic data for pretraining first, then fine-tune on real images for best competition performance.
- Keep this folder as the single place for experiment scripts and outputs.
