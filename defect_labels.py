from __future__ import annotations

UI_DEFECT_NONE = "无瑕疵"
UI_DEFECT_SCRATCH = "表面划痕"
UI_DEFECT_BUMP = "小凸起"
UI_DEFECT_DENT = "小凹陷"
UI_DEFECT_DUST = "灰尘污染"
UI_DEFECT_HAIR = "头发丝遮挡"

INTERNAL_DEFECT_BUMP = "小突起"

UI_SCRATCH_CONVEX = "凸起划痕"
UI_SCRATCH_CONCAVE = "凹陷划痕"
INTERNAL_SCRATCH_CONVEX = "突起划痕"

DEFECT_UI_TO_INTERNAL = {
    UI_DEFECT_BUMP: INTERNAL_DEFECT_BUMP,
}
DEFECT_INTERNAL_TO_UI = {
    INTERNAL_DEFECT_BUMP: UI_DEFECT_BUMP,
}

SCRATCH_UI_TO_INTERNAL = {
    UI_SCRATCH_CONVEX: INTERNAL_SCRATCH_CONVEX,
}
SCRATCH_INTERNAL_TO_UI = {
    INTERNAL_SCRATCH_CONVEX: UI_SCRATCH_CONVEX,
}

MODEL_LABEL_TO_UI = {
    "scratch": UI_DEFECT_SCRATCH,
    "bump": UI_DEFECT_BUMP,
    "dent": UI_DEFECT_DENT,
    "dust": UI_DEFECT_DUST,
    "hair": UI_DEFECT_HAIR,
}

UI_DEFECT_TO_MODEL = {
    UI_DEFECT_SCRATCH: "scratch",
    UI_DEFECT_BUMP: "bump",
    UI_DEFECT_DENT: "dent",
    UI_DEFECT_DUST: "dust",
    UI_DEFECT_HAIR: "hair",
}


def to_internal_defect_label(label: str) -> str:
    return DEFECT_UI_TO_INTERNAL.get(label, label)


def to_ui_defect_label(label: str) -> str:
    return DEFECT_INTERNAL_TO_UI.get(label, label)


def to_internal_scratch_label(label: str) -> str:
    return SCRATCH_UI_TO_INTERNAL.get(label, label)


def to_ui_scratch_label(label: str) -> str:
    return SCRATCH_INTERNAL_TO_UI.get(label, label)


def to_model_label_from_ui(label: str) -> str | None:
    return UI_DEFECT_TO_MODEL.get(to_ui_defect_label(label), UI_DEFECT_TO_MODEL.get(label))


def is_bump_or_dent_ui(label: str) -> bool:
    return to_ui_defect_label(label) in {UI_DEFECT_BUMP, UI_DEFECT_DENT}

