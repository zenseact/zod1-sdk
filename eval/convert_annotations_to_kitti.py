"""Converts dynamic object annotations to KITTI format."""
import argparse
import glob
import json
import os
from datetime import datetime
from os.path import join, basename
from typing import List, Callable

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from calibration import (
    load_calib_from_json,
    get_3d_transform_camera_lidar,
    rigid_transform_3d,
    transform_rotation,
)
from constants import TIME_FORMAT, SIZE, LOCATION, ROTATION
from plot_objects_on_image import ObjectAnnotationHandler

IMAGE_DIMS = np.array([3848, 2168])  # width, height

OCCLUSION_MAP = {
    "None": 0,
    "Light": 1,
    "Medium": 1,
    "Heavy": 2,
    "VeryHeavy": 2,
    "Undefined": 2,  # If undefined we assume the worst
}


def _parse_class(obj_properties):
    obj_cls = obj_properties["class"]
    if obj_cls not in ("VulnerableVehicle", "Vehicle", "Pedestrian"):
        # Remove Animals, Debris, Movers and any other unwanted classes
        return None
    elif obj_properties["unclear"] or obj_properties["object_type"] == "Inconclusive":
        # Ignore unclear and inconclusive objects
        obj_cls = "DontCare"
    elif obj_cls == "VulnerableVehicle":
        # Rename the VulnerableVehicle class to Cyclist to match KITTI
        obj_cls = "Cyclist"
        # Remove stuff without rider
        if obj_properties.get("with_rider", "True") == "False":
            return None
        # Ignore everything that's not a bicyclist or motorbicyclist
        elif obj_properties["object_type"] not in ("Bicycle", "Motorcycle"):
            obj_cls = "DontCare"
    elif obj_cls == "Vehicle":
        # Ignore more exotic vehicle classes (HeavyEquip, TramTrain, Other)
        if obj_properties["object_type"] not in ("Car", "Van", "Truck", "Trailer", "Bus"):
            obj_cls = "DontCare"
    elif obj_cls == "Pedestrian":
        # No special treatment for pedestrians
        pass
    return obj_cls


def _convert_to_kitti(
    objects: List[ObjectAnnotationHandler], yaw_func: Callable[[Quaternion], float]
) -> List[str]:
    kitti_annotation_lines = []
    for obj in objects:
        class_name = _parse_class(obj.properties)
        if class_name is None:
            continue  # discard object
        truncation, xmax, xmin, ymax, ymin = _parse_bbox_2d(obj.outer_points)
        if obj.marking3d is None:
            size, location, yaw, alpha = [0, 0, 0], [0, 0, 0], 0, 0
        else:
            size = obj.marking3d[SIZE][::-1]  # H,W,L not L,W,H
            location = obj.marking3d[LOCATION]  # x,y,z
            yaw = yaw_func(obj.marking3d[ROTATION])
            alpha = 0  # TODO: calculate this!
        if class_name != "DontCare" and "occlusion_ratio" not in obj.properties:
            print("Missing occlusion for obj: ", obj)
        kitti_obj = " ".join(
            map(
                str,
                [
                    class_name,
                    truncation,
                    OCCLUSION_MAP[obj.properties.get("occlusion_ratio", "Undefined")],
                    alpha,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    *size,
                    *location,
                    yaw,
                ],
            )
        )
        kitti_annotation_lines.append(kitti_obj)
    return kitti_annotation_lines


def _parse_bbox_2d(outer_points):
    xmin_nonclip, ymin_nonclip = np.min(outer_points, axis=0)
    xmax_nonclip, ymax_nonclip = np.max(outer_points, axis=0)
    xmin, ymin = np.clip([xmin_nonclip, ymin_nonclip], a_min=0, a_max=IMAGE_DIMS)
    xmax, ymax = np.clip([xmax_nonclip, ymax_nonclip], a_min=0, a_max=IMAGE_DIMS)
    new_area = (xmax - xmin) * (ymax - ymin)
    old_area = (xmax_nonclip - xmin_nonclip) * (ymax_nonclip - ymin_nonclip)
    truncation = 1 - new_area / old_area if old_area > 0.1 else 0
    return truncation, xmax, xmin, ymax, ymin


def _lidar_to_camera(objects, calib):
    for obj in objects:
        if obj.marking3d is None:
            continue
        transform = get_3d_transform_camera_lidar(calib)
        obj.marking3d[ROTATION] = transform_rotation(obj.marking3d[ROTATION], transform)
        obj.marking3d[LOCATION] = rigid_transform_3d(obj.marking3d[LOCATION], transform)
    return objects


def convert_annotation(calib_path, src_anno_pth, target_path):
    with open(src_anno_pth) as anno_file:
        src_anno = json.load(anno_file)
    vehicle, camera_name, time_str, id_ = basename(src_anno_pth.strip(".json")).split("_")
    id_ = int(id_)
    objects = ObjectAnnotationHandler.from_annotations(src_anno)
    objects = [obj[2] for obj in objects]

    # Convert objects from LIDAR to camera using calibration information
    frame_time = datetime.strptime(time_str, TIME_FORMAT)
    calib = load_calib_from_json(calib_path, vehicle, frame_time, camera_name)
    objects = _lidar_to_camera(objects, calib)
    # Write a KITTI-style annotation with obj in camera frame
    target_anno = _convert_to_kitti(objects, yaw_func=lambda rot: -rot.yaw_pitch_roll[0])
    with open(join(target_path, f"{id_:06d}.txt"), "w") as target_file:
        target_file.write("\n".join(target_anno))


def _parse_args():
    parser = argparse.ArgumentParser(description="Convert annotations to KITTI format")
    parser.add_argument("--dataset-dir", required=True, help="Root dataset directory")
    parser.add_argument("--target-dir", required=True, help="Output directory")
    return parser.parse_args()


def main():
    args = _parse_args()
    calib_path = join(args.dataset_dir, "calibration")
    source_path = join(args.dataset_dir, "annotations", "dynamic_objects")
    assert args.dataset_dir not in args.target_dir, "Do not write to the dataset"

    print("Looking up all source annotations...")
    source_anno_paths = glob.glob(f"{source_path}/*/*.json")

    # Create target directories
    os.makedirs(args.target_dir, exist_ok=True)

    for src_anno_pth in tqdm(source_anno_paths, desc="Converting annotations..."):
        try:
            convert_annotation(calib_path, src_anno_pth, args.target_dir)
        except Exception as err:
            print("Failed converting annotation: ", src_anno_pth, "with error:", str(err))
            raise


if __name__ == "__main__":
    main()
