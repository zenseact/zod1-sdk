"""Utils for annotation loading and processing."""
import glob
import os
import json
from typing import List, Sequence, Tuple
from pyquaternion import Quaternion

import numpy as np
import constants


def extract_info_from_annotation_filename(filename):
    """Extract info from annotations filename.

    Args:
        filename (str) : path to annotations filename or short representation
            (e.g golf_FC_2021-04-22T07:03:36.892736Z_dynamic_objects_0)

    Returns:
        anno_info (list) : list with info decoded from filename
            vehicle_id (str) : vehicle ID
            lidar_ts (int) : lidar pointcloud timestamp in unix format
            camera_ts (int) : camera frame timestamp in unix format
            camera_name (str) : camera name

    """
    parts = os.path.basename(filename).split(".")[0].split("_")
    if len(parts) != 5:
        raise ValueError(
            f"Filename {filename} does not fit the format "
            "<vehicle>_<camera_name>_<camera_unix_ts>_<annotation_project>_<sequence_id>"
        )
    try:
        vehicle_id = parts[0]
        camera_name = parts[1]
        camera_ts = int(parts[2])
        sequence_id = int(parts[4])
        return vehicle_id, camera_name, camera_ts, sequence_id
    except ValueError as error:
        raise ValueError(
            f"Filename {filename} does not fit the format "
            "<vehicle>_<camera_name>_<camera_unix_ts>_<annotation_project>_<sequence_id>"
        ) from error


def get_annotations_files(sequence_folder: str) -> dict:
    """Get all annotation files for a single sequence.

    Args:
        sequence_folder: path to folder containing data for a single frame sequence

    Returns:
        mapping from annotation project to file with annotations

    """

    def get_project_from_filename(filename: str) -> str:
        """Get annotation project from annotation file name."""
        return os.path.split(os.path.split(os.path.dirname(filename))[0])[1]

    anno_folder = sequence_folder.format(constants.ANNOTATIONS + "/*")
    files = glob.glob(os.path.join(anno_folder, "*.json"))
    return {get_project_from_filename(filename): filename for filename in files}


def read_anno_content(anno_file: str):
    """Read anno content."""
    with open(anno_file) as opened:
        content = json.load(opened)
    return content


class ObjectAnnotationHandler:
    """Class for representation of a single item for annotated dynamic and static objects."""

    __slots__ = ("item_class", "uuid", "camera_name", "outer_points", "marking3d", "properties")

    # pylint: disable=too-many-arguments
    def __init__(self, item_class, uuid, camera_name, outer_points, marking3d, properties):
        """Init.

        Args:
            item_class (str) : object's class
            uuid (str) : item's UUID identifier
            camera_name (str) : camera name where object is present
            outer_points (tuple[float]) : outer points representing 2D bounding box
            marking3d (dict[str, tuple[float]]) : location, orientation and size of 3D marking
            properties (dict[str, str]) : properties of the object

        """
        self.item_class = item_class
        self.uuid = uuid
        self.camera_name = camera_name
        self.outer_points = outer_points
        self.marking3d = marking3d
        self.properties = properties

    @classmethod
    def from_annotations(cls, anno_content):
        """Create list of ObjectAnnotationHandler from annotations file.

        Args:
            anno_content (dict) : annotation file content

        Returns:
            (generator[ObjectAnnotationHandler]) : generator of annotation items

        """

        def create_object(obj_info):
            """Create object from annotations file's content."""
            properties = obj_info[constants.PROPERTIES]
            item_class = (
                properties.get(constants.CLASS_KEY, constants.LANDMARKS)
                if properties
                else constants.LANDMARKS
            )
            filename = properties.get(constants.SCENE_ID, "") if properties else ""
            try:
                _, cam_name, _, _ = extract_info_from_annotation_filename(filename)
            except ValueError:
                cam_name = "Unknown"
            uuid = properties.get(constants.UUID_KEY, "") if properties else ""
            outer_points = obj_info[constants.GEOMETRY].get(constants.COORDS_KEY)
            if outer_points:
                outer_points = np.array(outer_points)
            if constants.MARKING_3D in properties:
                position = np.array(properties[constants.MARKING_3D][constants.COORDS_KEY])
                size = np.array([properties[dim] for dim in constants.SIZES])
                rotation = Quaternion(*[properties[rot] for rot in constants.ROTATIONS])
                marking3d = {
                    constants.LOCATION: position,
                    constants.SIZE: size,
                    constants.ROTATION: rotation,
                }
            else:
                marking3d = None
            return (
                item_class,
                uuid,
                cls(item_class, uuid, cam_name, outer_points, marking3d, properties),
            )

        objects = (create_object(obj_info) for obj_info in anno_content)
        return objects

    @property
    def corner_points(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get corner points."""
        corner_points = np.array(self.outer_points, dtype="float32").astype("int32")
        # draw corner points
        # for point, color in zip(corner_points, self.CORNER_COLORS):
        #     cv2.circle(img, tuple(point), self.POINT_RADIUS, color, thickness=-1)
        x_max = corner_points[:, 0].max()
        x_min = corner_points[:, 0].min()
        y_min = corner_points[:, 1].min()
        y_max = corner_points[:, 1].max()
        return (x_min, y_min), (x_max, y_max)

    def __repr__(self):
        """Create representation of item."""
        return "".join(
            [
                self.__class__.__name__,
                "(",
                r",".join(
                    map(
                        repr,
                        [
                            self.item_class,
                            self.uuid,
                            self.camera_name,
                            self.outer_points,
                            self.marking3d,
                            self.properties,
                        ],
                    )
                ),
                ")",
            ]
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
            properties = obj_info[constants.PROPERTIES]
            filename = properties.get(constants.SCENE_ID, "") if properties else ""
            try:
                _, cam_name, _, _ = extract_info_from_annotation_filename(filename)
            except ValueError:
                cam_name = "Unknown"
            uuid = properties.get(constants.UUID_KEY, "") if properties else ""
            polygon = obj_info[constants.GEOMETRY][constants.COORDS_KEY]
            return item_class, uuid, cls(item_class, uuid, cam_name, polygon, properties)

        objects = (create_object(project, obj_info) for obj_info in anno_content)
        return objects
