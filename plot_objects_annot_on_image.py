"""Module with tools to visualize annotated objects on the image."""
import glob
import os
from datetime import timedelta
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import constants
from colorlabeler import ColorLabeler
from annotation_utils import get_annotations_files, read_anno_content, ObjectAnnotationHandler


CAM_TIME_ERROR = timedelta(seconds=0.087)
FONT_TYPE = cv2.FONT_HERSHEY_COMPLEX


def apply_scale(values: Sequence[float], scale_factor: float) -> Sequence[float]:
    """Apply scale to values."""
    if scale_factor:
        return tuple(map(lambda x: scale_factor * x, values))
    return values


def calc_iou(box1_corners, box2_corners):
    """Calculate IoU between two boxes.

    Args:
        box1_corners (tuple[tuple[float]]) : left-top and right-bottom points of first box
        box2_corners (tuple[tuple[float]]) : left-top and right-bottom points of second box

    Returns:
        iou (float) : IoU metric

    """
    # select inner box corners
    inner_left_coord = max(box1_corners[0][0], box2_corners[0][0])
    inner_top_coord = max(box1_corners[0][1], box2_corners[0][1])
    inner_right_coord = min(box1_corners[1][0], box2_corners[1][0])
    inner_bottom_coord = min(box1_corners[1][1], box2_corners[1][1])

    # compute the area of intersection rectangle
    inter_area = abs(
        max((inner_right_coord - inner_left_coord, 0))
        * max((inner_bottom_coord - inner_top_coord), 0)
    )
    if inter_area == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    box1_area = abs(
        (box1_corners[0][0] - box1_corners[1][0]) * (box1_corners[0][1] - box1_corners[1][1])
    )
    box2_area = abs(
        (box2_corners[0][0] - box2_corners[1][0]) * (box2_corners[0][1] - box2_corners[1][1])
    )

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def visualize_object_2d_box(
    image: np.ndarray,
    do_object: ObjectAnnotationHandler,
    color: Tuple[int, int, int],
    scale_factor: float = None,
) -> np.ndarray:
    """Visualize 2D box of annotated object on the image."""
    left_up, right_bottom = do_object.corner_points
    left_up = apply_scale(left_up, scale_factor)
    right_bottom = apply_scale(right_bottom, scale_factor)
    cv2.rectangle(image, left_up, right_bottom, color=color, thickness=2)
    return image


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def visualize_object_properties(
    image: np.ndarray,
    left_up: Tuple[int, int],
    properties: dict,
    color: Tuple[int, int, int],
    properties_list: List[str],
    scale_factor: float = None,
    text_areas: list = None,
) -> np.ndarray:
    """Visualize properties values for object."""
    x_min, y_min = apply_scale(left_up, scale_factor)
    (width, height), _ = cv2.getTextSize("text", FONT_TYPE, constants.TEXT_SCALE, 2)
    # check text bounding box area
    text_area = [[x_min, y_min - height], [x_min, y_min]]
    text_lines = []
    for i, property_name in enumerate(properties_list):
        if property_name in properties:
            text = f"{property_name}:{properties[property_name]}"
        else:
            text = f"{property_name}:None"
        text_lines.append(text)
        (width, height), _ = cv2.getTextSize(text, FONT_TYPE, constants.TEXT_SCALE, 2)
        text_area[1][0] = max(text_area[1][0], x_min + width)
        text_area[1][1] = text_area[1][1] + height
    if text_areas and any(calc_iou(text_area, rec) for rec in text_areas):
        return image

    text_areas.append(text_area)

    # draw text
    for i, text in enumerate(text_lines):
        y = y_min + i * height
        if i == 0:
            (width, height), _ = cv2.getTextSize(text, FONT_TYPE, constants.TEXT_SCALE, 2)
            text_area = ((x_min, y - height), (x_min + width, y))
            cv2.rectangle(image, *text_area, color=(255, 255, 255), thickness=1)
        cv2.putText(image, text, (x_min, y), FONT_TYPE, constants.TEXT_SCALE, color, thickness=2)
    return image


# pylint: disable=too-many-locals
def visualize_annotated_objects_on_image(
    sequence_folder: str,
    object_properties_to_show: List[str] = None,
    scale_factor: float = None,
    project_name: str = None,
    images_folder: str = constants.BLURRED_IMAGES,
):
    """Visualize annotated objects on image.

    Args:
        sequence_folder: path to folder with data for a single sequence
        object_properties_to_show: list of properties names to be visualized over each object
        scale_factor: scale factor (in case if image was downsampled, all objects will be wrapped
            accordingly)
        project_name: name of annotations project
        images_folder: name of folder with images ("blurred_images" or "dnat_images")

    Return:
        image with objects visualized

    """
    anno_project_files = get_annotations_files(sequence_folder)
    if not anno_project_files:
        print(
            f"No annotation files found for sequence {sequence_folder.format(constants.ANNOTATIONS)}."
        )
        return None
    if project_name not in anno_project_files:
        print(
            f"No annotation file found for project {project_name} for sequence"
            f" {sequence_folder.format(constants.ANNOTATIONS)}: ",
            list(anno_project_files.keys()),
        )
        return None
    anno_file = anno_project_files[project_name]
    image_file = sorted(glob.glob(os.path.join(sequence_folder.format(images_folder), "*.png")))[1]
    image = cv2.imread(image_file)
    anno_content = read_anno_content(anno_file)
    anno_objects = list(ObjectAnnotationHandler.from_annotations(anno_content))
    object_colors = {obj[-1].item_class: i for i, obj in enumerate(anno_objects)}
    color_labeler = ColorLabeler(max_value=len(object_colors))
    object_colors = {
        obj_class: color_labeler.label_to_color(label)
        for label, obj_class in enumerate(object_colors)
    }
    text_areas = []
    for _, _, handler in anno_objects:
        color = object_colors[handler.item_class]
        left_up, _ = handler.corner_points
        image = visualize_object_2d_box(image, handler, color, scale_factor=scale_factor)
        image = visualize_object_properties(
            image,
            left_up,
            handler.properties,
            constants.TEXT_COLOR,
            object_properties_to_show,
            text_areas=text_areas,
            scale_factor=scale_factor,
        )
    return image
