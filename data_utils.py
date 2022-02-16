"""Utils for data loading."""
import glob
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List

import constants
import cv2
import h5py
import numpy as np
from pytz import utc


def gps_time_to_datetime(gps_time, gps_time_unit=constants.MICROSECONDS):
    """Convert time from gps format to timezone-aware datetime object.

    Args:
        gps_time (float): gps time in microseconds or seconds
        gps_time_unit (str): the timestamp unit, MICROSECONDS or SECONDS. Use DETECT
            to calculate the unit from the magnitude of the gps_time

    Return:
        datetime.datetime: timezone-aware datetime object

    Raises:
        AssertionError: if an invalid time unit specifier is provided
        RuntimeError: if there is a conflict between the supplied and detected input
            timestamp's precision

    """
    assert gps_time_unit in [constants.SECONDS, constants.MICROSECONDS, constants.DETECT]
    detected_time_unit = constants.MICROSECONDS if gps_time > 1e11 else constants.SECONDS

    if gps_time_unit not in (constants.DETECT, detected_time_unit):
        raise RuntimeError("Input timestamp does not appear to be in the format specificed!")

    if detected_time_unit == constants.SECONDS:
        gps_time *= constants.MICROSEC_PER_SEC

    return datetime(1980, 1, 6, tzinfo=utc) + timedelta(
        microseconds=gps_time - constants.LEAP_SECONDS * constants.MICROSEC_PER_SEC
    )


def _get_hdf5_datasets(filename: str) -> List[str]:
    """Get datasets from hdf5 file."""
    with h5py.File(filename, "r") as hdf5_file:
        # pylint: disable=unnecessary-comprehension
        datasets = [dataset for dataset in hdf5_file]
    return datasets


def _read_hdf5(path: str, dataset: str):
    """Read hdf5 data.

    Args:
        path : path to hdf5 file
        dataset : hdf5 dataset

    Returns:
        numpy.ndarray: hdf5 data

    """
    hdf5_file = h5py.File(path, "r")
    if dataset not in hdf5_file:
        return None
    return hdf5_file[dataset][:]


def load_oxts_from_dataset(folder: str) -> list:
    """Load OxTS data from dataset sequence folder.

    Args:
        folder : path to the folder containing data for a single dataset sequence

    """
    oxts_files = filter(
        lambda fln: constants.PREPROCESSED not in fln,
        glob.glob(os.path.join(folder, constants.HDF5_EXT)),
    )
    oxts_data = defaultdict(list)
    for oxts_file in oxts_files:
        datasets = _get_hdf5_datasets(oxts_file)
        for dataset in datasets:
            oxts_data[dataset].append(_read_hdf5(oxts_file, dataset))
    for dataset, oxts_datasets in oxts_data.items():
        oxts_data[dataset] = np.vstack(oxts_datasets).ravel()
    oxts_values = oxts_data[constants.OXTS_DATASET_KEY]
    return oxts_values


def load_images_from_dataset(folder: str) -> np.ndarray:
    """Load vision data from dataset sequence folder.

    Args:
        folder : path to the folder containing data for a single dataset sequence

    """
    images_files = sorted(glob.glob(os.path.join(folder, constants.PNG_EXT)))
    images = []
    for image_file in images_files:
        images.append(cv2.imread(image_file))
    return images, images_files


def load_lidar_from_dataset(folder: str, index: int = 0) -> np.ndarray:
    """Load Lidar data from dataset sequence folder.

    Args:
        folder : path to the folder containing data for a single dataset sequence
        index : index of pointcloud inside scan sequence, default 0 for single scan folder

    """
    lidar_file = sorted(glob.glob(os.path.join(folder, constants.NPY_EXT)))[index]
    pointcloud = np.load(lidar_file, allow_pickle=True)
    pointcloud = np.c_[pointcloud["x"], pointcloud["y"], pointcloud["z"], pointcloud["intensity"]]
    return pointcloud, lidar_file


def load_vehicle_data_from_dataset(folder: str) -> dict:
    """Load vehicle data from dataset sequence folder.

    Args:
        folder : path to the folder containing data for a single dataset sequence

    """
    vd_files = sorted(glob.glob(os.path.join(folder, constants.HDF5_EXT)))
    vehicle_data = defaultdict(list)
    for vd_file in vd_files:
        datasets = _get_hdf5_datasets(vd_file)
        for dataset in datasets:
            vehicle_data[dataset].append(_read_hdf5(vd_file, dataset))
    return vehicle_data
