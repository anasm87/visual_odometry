from enum import Enum
import cv2


class TrackerTypes(Enum):
    NONE = 0
    LUCASKANADE = 1  # lucas kanade pyramid tracker


class DetectorTypes(Enum):
    NONE = 0
    FAST = 1  # lucas kanade pyramid tracker


class DescriptorTypes(Enum):
    NONE = 0
    FAST = 1  # lucas kanade pyramid tracker


class DatasetType(Enum):
    NONE = 1
    KITTI = 2
    FOLDER = 3  # generic folder of pics


class FInfo(object):
    norm_type = dict()
    max_descriptor_distance = dict()  # max distance for the descriptor
    #
    norm_type[DescriptorTypes.NONE] = cv2.NORM_L2
    max_descriptor_distance[DescriptorTypes.NONE] = float('inf')


class VOstate(Enum):
    NOT_YET_INITIALIZED = 0
    INITIALIZED = 1


class KeyPointFilterTypes(Enum):
    NONE = 0
    SAT = 1  # number of features (keep the best N features: 'best' on the basis of the keypoint.response)
    KDT_NMS = 2  # Non-Maxima Suppression based on kd-tree
    SSC_NMS = 3  # Non-Maxima Suppression
    OCTREE_NMS = 4  # Distribute keypoints by using a octree
    GRID_NMS = 5  # NMS by using a grid

class GTType(Enum):
    NONE = 1
    SIMPLE = 2