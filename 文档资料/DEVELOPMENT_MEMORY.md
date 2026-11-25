# OpticalLab Defect Model Memory

## Required workflow

Before changing training data, training code, inference code, or model selection,
read this file first. After every material change, update the relevant sections
with the reason, files changed, validation result, and the next decision point.

## Deployment boundary

- Physics owns data construction, training, validation, and candidate weights.
- OpticalLab is the deployed frontend/backend consumer.
- The current OpticalLab baseline weight is:
  `dl_main_model_yolo/runs/train/yolo11s_small_target_1280_v1/weights/best.pt`.
- Do not change OpticalLab's model path until a Physics candidate beats this
  baseline on the fixed regression set and held-out validation data.

## Fixed OpticalLab Regression Set

Source images are imported to `dataset/regression/optical_lab_v1/images`.

| File | Confirmed label | Notes |
| --- | --- | --- |
| `1.png` | dust | Multiple isolated particle-like defects. |
| `2.png` | scratch | One connected zig-zag scratch. |
| `3.png` | hair | Multiple thin, discontinuous hair-like occlusions. |
| `4.png` | bump | Small bump. |
| `6.png` | dent | Three small dents. Replaces unlabelled `5.png`. |

The regression set is for final acceptance only. Never merge it into training,
validation, or test splits.

## Baseline Diagnosis: yolo11s_small_target_1280_v1

- Training completed on 2026-07-12, `imgsz=1280`, batch size 4.
- Its training configuration used `auto_augment=randaugment` and `erasing=0.4`.
  Both are unsuitable for thin hair/scratch and low-contrast fringe deformation.
- The normalized validation confusion matrix reports strong dust performance but
  excessive dust false positives on background, consistent with OpticalLab's
  observed misclassification.
- The model was trained before the current local-label rebuild. Its old validation
  score is not a valid acceptance metric for the rebuilt dataset.
- Fixed raw-image regression baseline at `conf=0.10`, `imgsz=1280`:
  - `1.png` (dust): top prediction `dust`, 20 dust boxes. Correct.
  - `2.png` (scratch): top prediction `dust`; 2 dust boxes and 1 scratch box.
    Incorrect because dust wins the displayed result.
  - `3.png` (hair): top prediction `dust`; 9 dust boxes and 1 hair box.
    Incorrect because dust wins the displayed result.
  - `4.png` (bump): top prediction `dust`, with 3 dust and 2 hair candidates.
    Incorrect.
  - `6.png` (dent): top prediction `dust`, with 5 dust and 1 hair candidates at
    regression confidence 0.10. OpticalLab's display threshold shows three dust
    boxes at 91-92%. Incorrect.
- Baseline report and visualisations:
  `dl_main_model_yolo/runs/regression/baseline_small_target_1280_v1_updated/best/report.json`.

## Data and Training Changes Implemented in Physics

- `dl_main_model_yolo/split_final_dataset.py`
  - Full-image weak labels are no longer written when localising a defect fails.
  - Invalid unlocalised samples are removed from the split rather than becoming
    full-image positives or background negatives.
  - Hair uses defect-versus-ideal-image local difference extraction when a mask
    is unavailable.
  - Per-image boxes are capped and valid minority training samples are balanced.
- `dl_main_model_yolo/train_final_split.py`
  - Default image size: 1024; batch size: 6.
  - Local-defect preset disables erasing and auto augmentation and limits spatial
    changes to preserve fringe geometry.
- Rebuilt training labels: scratch 2528, bump 1645, dent 1716, dust 6074,
  hair 5091. Full-image weak labels: 0.
- A one-epoch smoke training run completed successfully at
  `runs/train/defect_localization_smoke`.
- `dl_main_model_yolo/evaluate_regression.py` evaluates any candidate weight on
  the imported raw images and writes a JSON report plus annotated PNGs.
- `dl_main_model_yolo/audit_dataset_labels.py` creates deterministic per-class
  contact sheets before training. Its first audit found a major domain mismatch:
  the broad dataset contains Newton rings and multiple fringe colours, while the
  OpticalLab regression images are red wedge fringes.
- `split_final_dataset.py` now accepts `--sim-types` and wavelength bounds.
  The first deployment-focused candidate must use `--sim-types wedge` and
  `--min-wavelength-nm 620 --max-wavelength-nm 700` in a separate output folder.

## Next Decision Sequence

1. Inspect `runs/label_audit/train` after each rebuild. The initial contact sheets
   show that scratch/hair local boxes exist but may fragment curved defects; audit
   the red-wedge-only rebuild before deciding whether to tighten those rules.
2. Build the red-wedge dataset and train a candidate model named
   `yolo11s_opticallab_v2`; never overwrite the generic data folder or baseline.
3. Evaluate the candidate on the held-out split and the five-image regression set
   with `evaluate_regression.py`.
   Report per-class precision/recall, confusion matrix, boxes, and false positives.
4. Only then replace the OpticalLab model path with the candidate `best.pt`.

## Guardrails

- Do not use page screenshots as model inputs or labels; use the imported raw images.
- Do not trust overall mAP alone; dust dominance can conceal failure of other classes.
- Do not overwrite `yolo11s_small_target_1280_v1` or its weight file.
