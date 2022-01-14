"""Module for visualization of annotated polygons over image."""
import glob
import os
from itertools import chain
from typing import List, Tuple

import cv2
import numpy as np
from colorlabeler import ColorLabeler
from plot_objects_on_image import (
    extract_info_from_annotation_filename,
    get_annotations_files,
    read_anno_content,
    visualize_object_properties,
)

from constants import (
    ANNOTATIONS,
    BLURRED_IMAGES,
    COORDS_KEY,
    GEOMETRY,
    PROPERTIES,
    SCENE_ID,
    TEXT_COLOR,
    UUID_KEY,
)


class PolygonAnnotationHandler:
    """Base Class-handler representing info from annotate polygon item."""

    __slots__ = ("item_class", "uuid", "camera_name", "polygon", "properties")

    # pylint: disable=too-many-arguments
    def __init__(self, item_class, uuid, camera_name, polygon, properties):
        """Init.

        Args:
            item_class (str) : class this iter belongs to
            uuid (str) : annotation item uuid
            camera_name (str) : camera name
            polygon (list[list[float]]) : polygon coordinates of road painting item
            properties (dict[str,str]) : road painting item properties

        """
        self.item_class = item_class
        self.uuid = uuid
        self.camera_name = camera_name
        self.polygon = polygon
        self.properties = properties

    def polygon_to_numpy(self):
        """Return polygon coordinates as numpy array.

        Returns:
            (np.ndarray) : polygon coordinates

        """
        try:
            arr = np.asarray(self.polygon).astype("float32")
        except ValueError:
            arr = (
                np.asarray(list(chain.from_iterable(self.polygon))).reshape(-1, 2).astype("float32")
            )
        if arr.ndim > 2:
            arr = arr.squeeze()
        return arr

    def get_x_coordinates(self):
        """Return x-axis coordinates of polygon."""
        return self.polygon_to_numpy()[:, 0]

    def get_y_coordinates(self):
        """Return y-axis coordinates of polygon."""
        return self.polygon_to_numpy()[:, 1]

    @classmethod
    def from_annotations(
        cls, anno_content: dict, project: str
    ) -> List[Tuple[str, str, "PolygonAnnotationHandler"]]:
        """Create list of PolygonAnnotationHandler from annotations file.

        Args:
            anno_content (dict) : annotation file content

        Returns:
            (generator[PolygonAnnotationHandler]) : generator of annotation items

        """

        def create_object(item_class, obj_info):
            """Create object from annotations file's content."""
            properties = obj_info[PROPERTIES]
            filename = properties.get(SCENE_ID, "") if properties else ""
            try:
                _, cam_name, _, _ = extract_info_from_annotation_filename(filename)
            except ValueError:
                cam_name = "Unknown"
            uuid = properties.get(UUID_KEY, "") if properties else ""
            polygon = obj_info[GEOMETRY][COORDS_KEY]
            return item_class, uuid, cls(item_class, uuid, cam_name, polygon, properties)

        objects = (create_object(project, obj_info) for obj_info in anno_content)
        return objects


def visualize_polygon(
    image: np.ndarray,
    anno_handler: PolygonAnnotationHandler,
    color: Tuple[int, int, int],
    scale_factor: float = None,
    fill_color: Tuple[int, int, int] = None,
):
    """Visualize polygon on the image."""
    for poly in anno_handler.polygon:
        poly_array = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
        if scale_factor:
            poly_array *= scale_factor
        cv2.drawContours(image, [poly_array], -1, color, thickness=1)
        if fill_color:
            cv2.fillPoly(image, [poly_array], fill_color)
    return image


# pylint: disable=too-many-locals
def visualize_annotated_polygons_on_image(
    sequence_folder: str,
    object_properties_to_show: List[str] = None,
    scale_factor: float = None,
    project_name: str = None,
    images_folder: str = BLURRED_IMAGES,
):
    """Visualize annotated polygons on image.

    Args:
        sequence_folder: path to folder with data for a single sequence
        object_properties_to_show: list of properties names to be visualized over each object
        scale_factor: scale factor (in case if image was downsampled, all objects will be wrapped
            accordingly)
        project_name: name of annotations project
        images_folder: name of folder with images ("blurred_images" or "dnat_images")

    Return:
        image with polygons visualized

    """
    anno_project_files = get_annotations_files(sequence_folder)
    if not anno_project_files:
        print(f'No annotation files found for sequence {sequence_folder.format(ANNOTATIONS+"/*")}.')
        return None
    if project_name not in anno_project_files:
        print(
            f"No annotation file found for project {project_name} for sequence"
            f' {sequence_folder.format(ANNOTATIONS+"/*")}: ',
            list(anno_project_files.keys()),
        )
        return None
    anno_file = anno_project_files[project_name]
    image_file = sorted(glob.glob(os.path.join(sequence_folder.format(images_folder), "*.png")))[1]
    image = cv2.imread(image_file)
    anno_content = read_anno_content(anno_file)
    anno_objects = list(PolygonAnnotationHandler.from_annotations(anno_content, project_name))
    if project_name == "lane_markings":
        object_colors = {"lane_markings": 0, "road_paintings": 1}
    else:
        object_colors = {obj[-1].item_class: i for i, obj in enumerate(anno_objects)}
    color_labeler = ColorLabeler(max_value=len(object_colors))
    object_colors = {
        obj_class: color_labeler.label_to_color(label)
        for label, obj_class in enumerate(object_colors)
    }
    text_areas = []
    copy_image = image.copy()
    for _, _, handler in anno_objects:
        if project_name == "lane_markings":
            if handler.properties.get("class"):
                color = object_colors["lane_markings"]
            else:
                color = object_colors["road_paintings"]
        else:
            color = object_colors[handler.item_class]
        copy_image = visualize_polygon(copy_image, handler, color, scale_factor, fill_color=color)
        poly = handler.polygon_to_numpy()
        x_min = int(poly[:, 0].min())
        y_min = int(poly[:, 1].min())
        copy_image = visualize_object_properties(
            copy_image,
            (x_min, y_min),
            handler.properties,
            TEXT_COLOR,
            object_properties_to_show,
            text_areas=text_areas,
        )
    image = cv2.addWeighted(image, 0.5, copy_image, 0.5, 0)
    return image
