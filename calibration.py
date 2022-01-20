"""Module to work with dataset calibrations."""
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Type

import numpy as np
from pyquaternion import Quaternion

import constants


def load_calib_from_json(save_dir: str, vehicle: str, date: datetime, camera_name: str) -> dict:
    """Load calibration data from json file."""
    filename = os.path.join(
        save_dir,
        "_".join([vehicle, date.date().strftime(constants.DATE_FORMAT)]) + constants.JSON_EXT,
    )
    with open(filename) as opened:
        calib = json.load(opened)
    camera_calib = calib.get(camera_name)
    for key in constants.NUMPY_KEYS:
        camera_calib[key] = np.array(camera_calib[key])
    camera_calib[constants.INTRINSICS] = camera_calib[constants.INTRINSICS][:3, :3]
    camera_calib[constants.NAME] = camera_name
    return camera_calib


def check_dim_and_numaxes(data, dim):
    """Check that data has shape [n,dim] or [,dim].

    Args:
        data(ndarray(np.float)): Data to check dimensions and shape
        dim(int): Number of dimensions that data must represent.

    Raises:
        ValueError: If data is neither [dim,] nor [n,dim]

    """
    if data.shape == (dim,):
        return
    if data.ndim == 2:
        if data.shape[1] == dim:
            return
        raise ValueError(
            "Invalid shape ({}) of data to project to 2D. "
            "Expected (n,{}) data".format(data.shape, dim)
        )

    raise ValueError(
        "Invalid number of axes ({}) of data. " "Expected (n,{}) data".format(data.ndim, dim)
    )


def make_2_axes(data, dim=3):
    """Add 2nd axis if necessary.

    For 3D data, if it is represented with a one-axis vector with shape [3,], convert it to
    [1,3] shape. Otherwise, return unchanged.

    Args:
        data (ndarray(np.float)): [n,3] or [,3] data to add axis if needed
        dim(int): Number of dimensions that data should represent.

    Returns:
        ndarray(np.float): [n,3] data with content unchanged
        bool: whether data had shape [3,] originally

    """
    check_dim_and_numaxes(data, dim)
    one_axis_data = False
    if data.ndim == 1:
        data = data.reshape((1, -1))
        one_axis_data = True
    return data, one_axis_data


# pylint: disable=too-many-locals
def project_3d_to_2d_kannala(data, transform, camera_matrix, dist_coefs):
    """Project data from 3d coordinates to image plane using the Kanala model.

    Ref: J. Kannala and S. S. Brandt, "A generic camera model and calibration method for
    conventional, wide-angle, and fish-eye lenses," in IEEE Transactions on Pattern Analysis and
    Machine Intelligence, vol. 28, no. 8, pp. 1335-1340, Aug. 2006, doi: 10.1109/TPAMI.2006.153.

    Args:
        data (ndarray(np.float)): [n,3] data to transform
        transform (ndarray(np.float)): [4,4] transformation matrix of the frame where data is
            defined wrt the frame of the camera into whose plane data is to be projected
        camera_matrix (ndarray(np.float)): [3,4] camera matrix
        dist_coefs (ndarray(np.float)): [4,] distortion coeffs consisting of
            [k_1, k_2, k_3, k_4] coefficients (in that order)

    Returns:
        ndarray(np.float): [n,2] data in image plane

    """
    data, _ = make_2_axes(data)
    data_padded = np.hstack((data, np.ones((data.shape[0], 1))))
    data_transformed = (data_padded @ transform.T)[:, :-1]
    norm_data = np.linalg.norm(data_transformed[:, :2], axis=-1)
    radial = np.arctan2(norm_data, data_transformed[:, 2])
    radial2 = radial ** 2
    radial4 = radial2 ** 2
    radial6 = radial2 * radial4
    radial8 = radial4 ** 2
    distortion_angle = radial * (
        1
        + dist_coefs[0] * radial2
        + dist_coefs[1] * radial4
        + dist_coefs[2] * radial6
        + dist_coefs[3] * radial8
    )
    u_dist = distortion_angle * (data_transformed[:, 0] / norm_data)
    v_dist = distortion_angle * (data_transformed[:, 1] / norm_data)
    pos_u = camera_matrix[0, 0] * u_dist + camera_matrix[0, 2]
    pos_v = camera_matrix[1, 1] * v_dist + camera_matrix[1, 2]
    return np.stack((pos_u, pos_v), -1)


def rigid_transform_3d(data, transform):
    """Rigid transformation (translation + rotation) between two frames.

    Args:
        data (ndarray(np.float)): [n,3] data to transform
        transform (ndarray(np.float)): [4,4] transformation matrix of the frame where data is
            defined wrt the frame where data is to be transformed

    Returns:
        ndarray(np.float): [n,3] transformed data

    Raises:
        ValueError: If data is neither [3,] or [n,3]

    """
    data, one_axis_data = make_2_axes(data)
    data_padded = np.hstack((data, np.ones((data.shape[0], 1))))
    data_transformed = data_padded @ transform.T
    data_out = data_transformed[:, :-1]
    if one_axis_data:
        return data_out.flatten()
    return data_out


def invert_3d_transform(transform):
    """Invert a 3d rigid transform.

    Args:
        transform (ndarray(np.float)): [4,4] transform to invert.

    Returns:
        ndarray(np.float): [4,4] inverted transform

    """
    # pylint: disable=unbalanced-tuple-unpacking
    rot, tra = np.split(transform[:3, :], [3], 1)
    rot_tra = np.hstack((rot.T, -rot.T @ tra))
    return np.vstack((rot_tra, np.array([0, 0, 0, 1])))


def get_3d_transform_camera_lidar(calib: dict):
    """Get 3D transformation from lidar to camera."""
    t_refframe_to_frame = calib[constants.LIDAR_EXTRINSICS]
    t_refframe_from_frame = calib[constants.EXTRINSICS]

    t_from_frame_refframe = invert_3d_transform(t_refframe_from_frame)
    t_from_frame_to_frame = t_from_frame_refframe @ t_refframe_to_frame

    return t_from_frame_to_frame


def transform_rotation(rotation: Quaternion, transform: np.ndarray):
    """Transform the rotation between two frames defined by the transformation."""
    return Quaternion(matrix=transform[:3, :3].T) * rotation


class CameraInfo(ABC):
    """Class to handle camera info."""

    CAMERA_TYPE = None
    __slots__ = "name", "camera_type", "width", "height", "intrinsics", "distortion_coefs", "xi"

    # pylint: disable=too-many-arguments
    def __init__(self, name, camera_type, width, height, intrinsics, distortion_coefs, xi):
        """Init."""
        self.name = name
        self.camera_type = camera_type
        self.width = width
        self.height = height
        self.intrinsics = intrinsics
        self.distortion_coefs = distortion_coefs
        self.xi = xi

    @abstractmethod
    def project(self, lidar_points: np.ndarray, crop: bool) -> np.ndarray:
        """Calculate lidar points positions on the image plane."""

    def _create_xyd_array(
        self, img_plane_positions: np.ndarray, lidar_points: np.ndarray, crop: bool
    ) -> np.ndarray:
        if crop:
            masks = np.ones((lidar_points.shape[0], 2), dtype=bool)
            masks[:, 0] = np.logical_and(
                img_plane_positions[:, 0] > -constants.HALF_PIXEL_SIZE,
                img_plane_positions[:, 0] < self.width - constants.HALF_PIXEL_SIZE,
            )
            masks[:, 1] = np.logical_and(
                img_plane_positions[:, 1] > -constants.HALF_PIXEL_SIZE,
                img_plane_positions[:, 1] < self.height - constants.HALF_PIXEL_SIZE,
            )
            in_img = masks.all(axis=1)

            img_plane_positions = img_plane_positions[in_img, :]

            xyd_array = np.concatenate([img_plane_positions, lidar_points[in_img, 2:3]], axis=1)
        else:
            xyd_array = np.concatenate([img_plane_positions, lidar_points[:, 2:3]], axis=1)
        return xyd_array

    @staticmethod
    def create_instance_from_config(config_info: dict) -> Type["CameraInfo"]:
        """Create instance of CameraInfo from config parameters."""
        camera_type = config_info.get(constants.CAMERA_TYPE)
        assert camera_type, "The camera type is absent in config data"
        for camera_info_class in CameraInfo.__subclasses__():
            if camera_info_class.CAMERA_TYPE == camera_type:
                return camera_info_class.create_from_config(config_info)
        raise ValueError(f"No CameraInfo instance can be created for camera type {camera_type}")

    @classmethod
    @abstractmethod
    def create_from_config(cls, config: dict) -> Type["CameraInfo"]:
        """Create instance from config."""


class KannalaCameraInfo(CameraInfo):
    """Class to handle camera info for Cannala camera type."""

    CAMERA_TYPE = constants.KANNALA

    # pylint: disable=too-many-arguments
    def __init__(self, name, width, height, intrinsics, distortion_coefs):
        """Init."""
        super().__init__(name, constants.KANNALA, width, height, intrinsics, distortion_coefs, None)

    def project(self, lidar_points: np.ndarray, crop: bool) -> np.ndarray:
        """Calculate lidar points positions on the image plane."""
        no_transform = np.eye(4)
        img_plane_positions = project_3d_to_2d_kannala(
            lidar_points, no_transform, self.intrinsics, self.distortion_coefs
        )
        return self._create_xyd_array(img_plane_positions, lidar_points, crop)

    @classmethod
    def create_from_config(cls, config: dict) -> "KannalaCameraInfo":
        """Create instance from config."""
        width, height = config[constants.DIMENSIONS]
        return KannalaCameraInfo(
            config["name"],
            width,
            height,
            config[constants.INTRINSICS],
            config[constants.DISTORTION],
        )
