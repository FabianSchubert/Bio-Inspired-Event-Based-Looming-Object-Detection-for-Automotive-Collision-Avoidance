import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_DATA = os.path.abspath(os.path.join(ROOT_DIR, "../data/"))

BOXES_DTYPE = [
    ("ts", "<u8"),
    ("x", "<f4"),
    ("y", "<f4"),
    ("w", "<f4"),
    ("h", "<f4"),
    ("class_id", "u1"),
    ("confidence", "<f4"),
    ("track_id", "<u4"),
]

BOXES_YOLO_DTYPE = [
    ("x", "<f4"),
    ("y", "<f4"),
    ("w", "<f4"),
    ("h", "<f4"),
    ("class_id", "u1"),
]

EVENTS_DTYPE = [("t", "<u4"), ("x", "<u2"), ("y", "<u2"), ("p", "<u2")]
