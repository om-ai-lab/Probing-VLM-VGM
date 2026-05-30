"""ScanNet 20 / 200 class taxonomies — canonical valid IDs + labels.

Sources (verbatim from the official ScanNet repo):
  - ScanNet20:  https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/util.py
                Uses NYU40 IDs (column 5 of scannetv2-labels.combined.tsv).
  - ScanNet200: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/ScanNet200/scannet200_constants.py
                Uses raw ScanNet IDs (column 1 of scannetv2-labels.combined.tsv).

The 200-class set is a strict superset of the 20-class set's semantics but uses a
DIFFERENT id namespace (raw vs nyu40), so the per-class lookups can't share a
table. Use `build_raw_to_class_map(tsv_path, num_classes)` to construct the
runtime mapping `raw_category_str → class_index ∈ [0, num_classes)`.
"""
from __future__ import annotations

from typing import Dict, Tuple

# ---------------------------------------------------------------------- #
# ScanNet20 (NYU40 id-based)
# ---------------------------------------------------------------------- #
VALID_CLASS_IDS_20: Tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39,
)

CLASS_LABELS_20: Tuple[str, ...] = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture",
)
assert len(VALID_CLASS_IDS_20) == 20 and len(CLASS_LABELS_20) == 20


# ---------------------------------------------------------------------- #
# ScanNet200 (raw ScanNet id-based)
# ---------------------------------------------------------------------- #
VALID_CLASS_IDS_200: Tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23,
    24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45,
    46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66,
    67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87,
    88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
    107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132,
    134, 136, 138, 139, 140, 141, 145, 148, 154, 155, 156, 157, 159, 161,
    163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202,
    208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283,
    286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408,
    417, 488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164,
    1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176,
    1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189,
    1190, 1191,
)

CLASS_LABELS_200: Tuple[str, ...] = (
    "wall", "chair", "floor", "table", "door", "couch", "cabinet", "shelf",
    "desk", "office chair", "bed", "pillow", "sink", "picture", "window",
    "toilet", "bookshelf", "monitor", "curtain", "book", "armchair",
    "coffee table", "box", "refrigerator", "lamp", "kitchen cabinet",
    "towel", "clothes", "tv", "nightstand", "counter", "dresser", "stool",
    "cushion", "plant", "ceiling", "bathtub", "end table", "dining table",
    "keyboard", "bag", "backpack", "toilet paper", "printer", "tv stand",
    "whiteboard", "blanket", "shower curtain", "trash can", "closet",
    "stairs", "microwave", "stove", "shoe", "computer tower", "bottle",
    "bin", "ottoman", "bench", "board", "washing machine", "mirror",
    "copier", "basket", "sofa chair", "file cabinet", "fan", "laptop",
    "shower", "paper", "person", "paper towel dispenser", "oven", "blinds",
    "rack", "plate", "blackboard", "piano", "suitcase", "rail", "radiator",
    "recycling bin", "container", "wardrobe", "soap dispenser", "telephone",
    "bucket", "clock", "stand", "light", "laundry basket", "pipe",
    "clothes dryer", "guitar", "toilet paper holder", "seat", "speaker",
    "column", "bicycle", "ladder", "bathroom stall", "shower wall", "cup",
    "jacket", "storage bin", "coffee maker", "dishwasher",
    "paper towel roll", "machine", "mat", "windowsill", "bar", "toaster",
    "bulletin board", "ironing board", "fireplace", "soap dish",
    "kitchen counter", "doorframe", "toilet paper dispenser", "mini fridge",
    "fire extinguisher", "ball", "hat", "shower curtain rod", "water cooler",
    "paper cutter", "tray", "shower door", "pillar", "ledge", "toaster oven",
    "mouse", "toilet seat cover dispenser", "furniture", "cart",
    "storage container", "scale", "tissue box", "light switch", "crate",
    "power outlet", "decoration", "sign", "projector", "closet door",
    "vacuum cleaner", "candle", "plunger", "stuffed animal", "headphones",
    "dish rack", "broom", "guitar case", "range hood", "dustpan",
    "hair dryer", "water bottle", "handicap bar", "purse", "vent",
    "shower floor", "water pitcher", "mailbox", "bowl", "paper bag",
    "alarm clock", "music stand", "projector screen", "divider",
    "laundry detergent", "bathroom counter", "object", "bathroom vanity",
    "closet wall", "laundry hamper", "bathroom stall door", "ceiling light",
    "trash bin", "dumbbell", "stair rail", "tube", "bathroom cabinet",
    "cd case", "closet rod", "coffee kettle", "structure", "shower head",
    "keyboard piano", "case of water bottles", "coat rack",
    "storage organizer", "folded chair", "fire alarm", "power strip",
    "calendar", "poster", "potted plant", "luggage", "mattress",
)
assert len(VALID_CLASS_IDS_200) == 200 and len(CLASS_LABELS_200) == 200


# ---------------------------------------------------------------------- #
# TSV mapping builder
# ---------------------------------------------------------------------- #
def build_raw_to_class_map(tsv_path: str, num_classes: int) -> Dict[str, int]:
    """Parse scannetv2-labels.combined.tsv and return raw_category → class_idx.

    Args:
        tsv_path:     path to scannetv2-labels.combined.tsv (tab-separated)
        num_classes:  20 or 200 — selects which taxonomy to use.

    Returns:
        dict mapping raw_category (str, e.g. "kitchen counter") to class
        index in [0, num_classes). Raw categories that fall outside the
        chosen taxonomy are absent from the dict.

    Notes:
        The TSV's column-1 "id" and column-5 "nyu40id" are different
        namespaces. ScanNet20 keys on nyu40id; ScanNet200 keys on the raw
        ScanNet id. We always read both columns to keep this dispatch
        local instead of split across callsites.
    """
    if num_classes == 20:
        valid_ids = VALID_CLASS_IDS_20
        id_col_index = 4   # nyu40id (0-indexed column 5)
    elif num_classes == 200:
        valid_ids = VALID_CLASS_IDS_200
        id_col_index = 0   # raw scannet id (0-indexed column 1)
    else:
        raise ValueError(f"num_classes must be 20 or 200, got {num_classes}")

    valid_id_to_idx = {raw_id: i for i, raw_id in enumerate(valid_ids)}

    raw_to_idx: Dict[str, int] = {}
    with open(tsv_path, "r") as f:
        header = f.readline()  # skip header
        del header
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            try:
                raw_id = int(cols[id_col_index])
            except ValueError:
                continue
            raw_category = cols[1]
            if raw_id in valid_id_to_idx:
                raw_to_idx[raw_category] = valid_id_to_idx[raw_id]
    return raw_to_idx


def get_class_labels(num_classes: int) -> Tuple[str, ...]:
    if num_classes == 20:
        return CLASS_LABELS_20
    if num_classes == 200:
        return CLASS_LABELS_200
    raise ValueError(f"num_classes must be 20 or 200, got {num_classes}")
