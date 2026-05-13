# defect_ai_module

A standalone deep-learning module for defect inference.

## Goals
- Keep model inference independent from UI framework.
- Reuse the same service in desktop and future web backends.
- Minimize merge conflicts when platform code is updated by teammates.

## Main API
- `DefectAIService`
  - `set_model_path(model_path)`
  - `predict_image(image_path, conf=0.25, iou=0.7, device="0", save_visualization=True)`

## Quick Usage
```python
from defect_ai_module import DefectAIService

service = DefectAIService()
result = service.predict_image("demo.png")

if result.top_prediction:
    print(result.top_prediction.label_cn, result.top_prediction.confidence)
print(result.output_image)
```

## Integration Notes
- Existing desktop wrapper remains in `OpticalDefectExperiment/defect_inference.py`.
- For web migration, call `DefectAIService` directly from API routes.
