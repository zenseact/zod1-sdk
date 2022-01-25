import json
import math
import os
from typing import List

import cv2

from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

OBJECT_CATEGORIES = {
    "Vehicle": 0,
    "Cyclist": 1,
    "Pedestrian": 2,
}

DATASET_TRAIN = "zen_2dod_train"
DATASET_VAL = "zen_2dod_val"


def _read_objs(path):
    objs = []
    with open(path, "r") as anno_file:
        for obj in anno_file:
            obj = obj.split(' ')
            category_id = OBJECT_CATEGORIES.get(obj[0], None)
            if category_id is not None:
                objs.append({
                    "bbox": [float(val) for val in obj[4:8]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": category_id,
                })
    return objs


def get_dataset_dicts(root_path: str, test: bool) -> List[dict]:
    # First construct a map from id to image path
    datalist_path = os.path.join(root_path, ('test.json' if test else 'train.json'))
    with open(datalist_path) as datalist_file:
        datalist = json.load(datalist_file)
    # Then read annotations and construct dataset dict
    frames = []
    print("Loading annotations...")
    for frame_id, frame_paths in datalist.items():
        frame_id = int(frame_id)
        record = {
            "file_name": sorted(frame_paths["blurred_imgs"])[1],
            "image_id": frame_id,
            "height": 2168,
            "width": 3848,
            "annotations": _read_objs(frame_paths["anno"][0]),
        }
        frames.append(record)
    return frames


def register_detectron(
    base_dir: str,
    split: int = 0,
    num_splits: int = 3,
):
    # Split into train and val datasets
    dataset_dict = get_dataset_dicts(base_dir, test=False)
    frames_per_split = math.ceil(len(dataset_dict) / num_splits)
    val_start, val_end = split * frames_per_split, (split + 1) * frames_per_split
    train_dataset = dataset_dict[0:val_start] + dataset_dict[val_end:]
    val_dataset = dataset_dict[val_start: val_end]

    # Register datasets
    DatasetCatalog.register(DATASET_TRAIN, lambda: train_dataset)
    DatasetCatalog.register(DATASET_VAL, lambda: val_dataset)
    MetadataCatalog.get(DATASET_TRAIN).thing_classes = list(OBJECT_CATEGORIES.keys())
    MetadataCatalog.get(DATASET_VAL).thing_classes = list(OBJECT_CATEGORIES.keys())


if __name__ == "__main__":
    # This code is only for debugging / visualization.
    DATASET_ROOT = ''  # insert the dataset root path here
    register_detectron(DATASET_ROOT, split=0, num_splits=3)
    dataset = DatasetCatalog.get(DATASET_TRAIN)
    for d in dataset:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1], scale=1, metadata=MetadataCatalog.get(DATASET_TRAIN)
        )
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("image", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        break
