"""This script runs a trained detector and saves the detections in KITTI format."""
import argparse
import glob
import json
import math
import os

import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from tqdm import tqdm

import examples.object_detection_2d.detectron_dataset as detectron_dataset
from examples.object_detection_2d.train import MODEL_PATH

setup_logger()

CAT_ID_TO_CLASS_NAME = {v: k for k, v in detectron_dataset.OBJECT_CATEGORIES.items()}


def get_predictor(weight_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 2048
    cfg.MODEL.WEIGHTS = weight_path
    return DefaultPredictor(cfg)


def get_split_datalist(split: int, num_splits: int, dataset_root: str):
    """Get the (validation) datalist for a given split (fold)."""
    with open(os.path.join(dataset_root, "train.json")) as datalist_file:
        datalist = json.load(datalist_file)

    # Slice out the validation frames for the given split
    frames_per_split = math.ceil(len(datalist) / num_splits)
    val_start = split * frames_per_split
    val_end = (split + 1) * frames_per_split
    print(f"Using validation data from [{val_start}:{val_end}]")
    return list(datalist.items())[val_start:val_end]


def get_test_datalist(dataset_root: str):
    with open(os.path.join(dataset_root, "test.json")) as datalist_file:
        datalist = json.load(datalist_file)
    return list(datalist.items())


def get_unlabeled_datalist(dataset_root: str):
    with open(os.path.join(dataset_root, "unlabeled.json")) as datalist_file:
        datalist = json.load(datalist_file)
    return list(datalist.items())


def save_dets_for_sequence(predictor, id_, paths, detection_base_path):
    assert len(paths["blurred_imgs"]) == 3, f"Too many imgs in {paths}"
    det_path = os.path.join(detection_base_path, id_)
    os.makedirs(det_path, exist_ok=True)
    for path in paths["blurred_imgs"]:
        im = cv2.imread(path)
        instances = predictor(im)["instances"].to("cpu")
        det_file_name = os.path.basename(path).replace(".png", ".txt")
        with open(os.path.join(det_path, det_file_name), "w") as det_file:
            for det_idx in range(len(instances)):
                det_file.write(
                    f" ".join(
                        map(
                            str,
                            [
                                CAT_ID_TO_CLASS_NAME[int(instances.pred_classes[det_idx])],
                                -1,  # truncation
                                -1,  # occlusion
                                -1,  # alpha
                                *instances.pred_boxes.tensor[det_idx].numpy(),
                                *(-1, -1, -1),  # size
                                *(-1, -1, -1),  # location
                                -1,  # yaw
                                instances.scores[det_idx].numpy(),  # score
                            ],
                        )
                    )
                )
                det_file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, help="Root path of the dataset")
    parser.add_argument("--output-path", required=True, help="Path where detections will be stored")
    parser.add_argument("--split", default=0, type=int)
    parser.add_argument("--num-splits", default=3, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--unlabeled", action="store_true")
    return parser.parse_args()


def main(args):
    predictor = get_predictor(f"/workspace/d2_kitti_2dod/split{args.split}/model_final.pth")
    if args.test:
        print("Running inference on test data")
        datalist = get_test_datalist(args.dataset_path)
    elif args.unlabeled:
        print("Running inference on unlabeled data")
        datalist = get_unlabeled_datalist(args.dataset_path)
    else:
        print(f"Running inference on the validation split of fold {args.split}/{args.num_splits}")
        datalist = get_split_datalist(args.split, args.num_splits, args.dataset_dir)

    detection_path = os.path.join(args.dataset_dir, "detections", "camera")
    for id_, paths in tqdm(datalist):
        save_dets_for_sequence(predictor, id_, paths, detection_path)


if __name__ == "__main__":
    args = parse_args()
    print("Command Line Args:", args)
    main(args)
