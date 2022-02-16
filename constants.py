"""Constants for dataset visualization kit."""

# absolute path to data folder
DATA_ROOT = ""
# name of the folder containing OxTS data in dataset for a single frame sequence
OXTS_FOLDER = "oxts"
# extension of file containing OxTS data
HDF5_EXT = "*.hdf5"
# dataset key in hdf5 file for OxTS data
OXTS_DATASET_KEY = "oxts"
# keys in OxTS data
TIMESTAMP_KEY = "timestamp"
SECONDS = "seconds"
LONGITUDE = "posLon"
LATITUDE = "posLat"
# possible choices for GPS timestamp conversion
MICROSECONDS = "microseconds"
DETECT = "detect"
MICROSEC_PER_SEC = 1e6
NANOSEC_PER_MICROSEC = 1e3
LEAP_SECONDS = 18
BLUE_COLOR = "blue"
PNG_EXT = "*.png"
NPY_EXT = "*.npy"
VISION_FOLDER = "vision"
LIDAR_FOLDER = "lidar"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"
EXTRINSICS = "extrinsics"
INTRINSICS = "intrinsics"
DISTORTION = "distortion"
DYNAMIC_OBJECTS = "dynamic_objects"
LANE_MARKINGS = "lane_markings"
STATIC_OBJECTS = "static_objects"
TRAFFIC_SIGNS = "traffic_signs"
EGO_ROAD = "ego_road"
PREPROCESSED = "_preprocessed"

JSON_EXT = ".json"
LIDAR_EXTRINSICS = "lidar_extrinsics"
CAMERA_TYPE = "camera_type"
DIMENSIONS = "image_dimensions"
XI = "xi"
NAME = "name"
DATE_FORMAT = "%Y-%m-%d"
FOV = "field_of_view"

NUMPY_KEYS = [INTRINSICS, EXTRINSICS, DISTORTION, LIDAR_EXTRINSICS]
HALF_PIXEL_SIZE = 0.5
KANNALA = "kannala"

# Plot GPS constants
DEFAULT_COLOR = "red"
DEFAULT_SIZE = 1
MAPS_STYLE = "open-street-map"
SIZE_MAX = 7
OPACITY_LEVEL = 1
DB_DATE_STRING_FORMAT_W_MICROSECONDS = "%Y-%m-%dT%H:%M:%S.%fZ"
DB_DATE_STRING_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
DEFAULT_COL_VALUES = {
    "pitchMissalignment": (0, 0, 0, b"Radians"),
    "headingMissalignment": (0, 0, 0, b"Radians"),
}
OXTS_OPTIONAL_COLS = [*DEFAULT_COL_VALUES.keys()]
OXTS_COLS = [
    "undulation",
    "timestamp",
    "posLat",
    "posLon",
    "posAlt",
    "heading",
    "pitch",
    "roll",
    "velForward",
    "velDown",
    "velLateral",
    "leapSeconds",
]
ECEF_XYZ = ["ecef_x", "ecef_y", "ecef_z"]

# Objects plot constants
UNCLEAR = "unclear"
FEATURES = "features"
GEOMETRY = "geometry"
PROPERTIES = "properties"
CLASS_KEY = "class"
UUID_KEY = "annotation_uuid"
COORDS_KEY = "coordinates"
INCONCLUSIVE = "Inconclusive"
COORDINATES = "coordinates"
SCENE_ID = "scene_id"
LANDMARKS = "Landmarks"
MARKING_3D = "location_3d"
X = "X"
Y = "Y"
Z = "Z"
L = "size_3d_length"
W = "size_3d_width"
H = "size_3d_height"
QW = "orientation_3d_qw"
QX = "orientation_3d_qx"
QY = "orientation_3d_qy"
QZ = "orientation_3d_qz"
POSITIONS = (X, Y, Z)
SIZES = (L, W, H)
ROTATIONS = (QW, QX, QY, QZ)
LOCATION = "Location"
SIZE = "Size"
ROTATION = "Rotation"
ORIENTATION = "Orientation"
TEXT_COLOR = (0, 0, 0)
TEXT_SCALE = 0.9
ANNOTATION = "annotation"
VISION = "vision"
ANNOTATIONS = "annotations"
ORIGINAL_IMAGES = "original_images"
BLURRED_IMAGES = "blurred_images"
DNAT_IMAGES = "dnat_images"

# Dynamic Object classes for BEV visualization
DO_CLASSES = ("Vehicle", "VulnerableVehicle", "Pedestrian", "Animal")
# Static objects classes for BEV visualization
SO_CLASSES = (
    "TrafficSign",
    "PoleObject",
    "TrafficGuide",
    "TrafficBeacon",
    "TrafficSignal",
    "DynamicBarrier",
    "Inconclusive",
)