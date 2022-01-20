"""This script takes the provided raw detections and converts them to a evaluation-friendly format."""
import argparse
import glob
import json
import os
import shutil
from datetime import datetime

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from calibration import (
    get_3d_transform_camera_lidar,
    transform_rotation,
    rigid_transform_3d,
    load_calib_from_json,
)
from constants import TIME_FORMAT


def _get_train_ids(dataset_root: str):
    with open(os.path.join(dataset_root, "train.json")) as datalist_file:
        datalist = json.load(datalist_file)
    return list(datalist.keys())


def _get_test_ids(dataset_root: str):
    with open(os.path.join(dataset_root, "test.json")) as datalist_file:
        datalist = json.load(datalist_file)
    return list(datalist.keys())


def _prepare_camera_detection(dataset_dir, camera_output_dir, id_):
    """Copy camera detection."""
    camera_detection_path = os.path.join(dataset_dir, "detections", "camera")
    camera_new_path = os.path.join(camera_output_dir, f"{int(id_):06d}.txt")
    camera_old_path = list(sorted(glob.glob(os.path.join(camera_detection_path, id_, "*"))))[1]
    shutil.copy(camera_old_path, camera_new_path)
    return camera_old_path


def _prepare_lidar_detection(dataset_dir, lidar_output_dir, id_, camera_detection_path):
    """Prepare lidar detection file, including coordinate transformation."""
    # Find the lidar frame closest to the annotated camera timestamp
    lidar_detection_path = os.path.join(dataset_dir, "detections", "lidar")
    camera_timestamp = datetime.strptime(
        os.path.basename(camera_detection_path).split("_")[2], TIME_FORMAT
    )
    lidar_paths = list(sorted(glob.glob(os.path.join(lidar_detection_path, id_, "*"))))
    lidar_timestamps = [
        datetime.strptime(os.path.basename(lidar_path).split("_")[1], TIME_FORMAT)
        for lidar_path in lidar_paths
    ]
    diffs = [abs((camera_timestamp - lid_time).total_seconds()) for lid_time in lidar_timestamps]
    _, min_idx = min((diff, idx) for (idx, diff) in enumerate(diffs))
    lidar_path = lidar_paths[min_idx]

    # Load calibration
    calib_path = os.path.join(dataset_dir, "calibration")
    # example name: golf_FC_2021-04-22T07:03:36.859402Z_0.txt
    vehicle, camera_name, _, _ = os.path.basename(camera_detection_path).split("_")
    calib = load_calib_from_json(calib_path, vehicle, camera_timestamp, camera_name)

    # Modify the detections
    new_lidar_lines = []
    with open(lidar_path) as lidar_file:
        for line in lidar_file:
            line = line.split(" ")
            pos = np.array([float(val) for val in line[11:14]])
            rot = Quaternion(axis=[0.0, 0.0, 1.0], angle=float(line[14]))
            transform = get_3d_transform_camera_lidar(calib)
            # The transformed rotation will have a 90deg roll and the rotation around camera y is -yaw
            new_rot = str(-transform_rotation(rot, transform).yaw_pitch_roll[0])
            new_pos = list(map(str, rigid_transform_3d(pos, transform)))
            line[11:14], line[14] = new_pos, new_rot
            # Change ymax to pass the 25px eval height check (ymin is 0)
            line[7] = "25.1"
            new_lidar_lines.append(line)

    # Write new detection file
    new_path = os.path.join(lidar_output_dir, f"{int(id_):06d}.txt")
    with open(new_path, "w") as new_file:
        for line in new_lidar_lines:
            new_file.write(" ".join(line))

    # TODO: merge camera and lidar to one file for joint eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help='root dataset directory')
    parser.add_argument("--tmp-det-dir", default="/tmp/detections")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def main(args):
    if args.test:
        print("Preparing eval on test data")
        ids = _get_test_ids(args.dataset_dir)
    else:
        print(f"Preparing eval on the train data")
        ids = _get_train_ids(args.dataset_dir)

    # Prepare paths
    camera_output_dir = os.path.join(args.tmp_det_dir, "camera", "data")
    lidar_output_dir = os.path.join(args.tmp_det_dir, "lidar", "data")
    os.makedirs(camera_output_dir, exist_ok=True)
    os.makedirs(lidar_output_dir, exist_ok=True)

    for id_ in tqdm(ids):
        camera_detection_path = _prepare_camera_detection(args.dataset_dir, camera_output_dir, id_)
        _prepare_lidar_detection(args.dataset_dir, lidar_output_dir, id_, camera_detection_path)

    print(f"stored the relevant detections in {args.tmp_det_dir}")


if __name__ == "__main__":
    args = parse_args()
    print("Command Line Args:", args)
    main(args)
