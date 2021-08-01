from data_types import *
import cv2
import numpy as np
import math

kNumLevelsInitSigma = 40
kSigmaLevel0 = 1.0
kNumFeatures = 2000
kFASTKeyPointSizeRescaleFactor = 4


#  best features to keep
def sat_num_features(keypoints, des=None, num_features=kNumFeatures):
    if len(keypoints) > num_features:
        # keep the features with the best response
        if des is None:
            kps = sorted(keypoints, key=lambda x: x.response, reverse=True)[:num_features]
        else:
            # sort by score to keep highest score features
            # print('sat with des')
            order = np.argsort([kp.response for kp in keypoints])[::-1][:num_features]  # [::-1] is for reverse order
            kps = np.array(keypoints)[order]
            des = np.array(des)[order]
    return kps, des


def hamming_distance(a, b):
    return np.count_nonzero(a != b)


def hamming_distances(a, b):
    return np.count_nonzero(a != b, axis=1)


def l2_distance(a, b):
    return np.linalg.norm(a.ravel() - b.ravel())


def l2_distances(a, b):
    return np.linalg.norm(a - b, axis=-1, keepdims=True)


def import_f(f_module, f_name, method=None):
    try:
        im_module = __import__(f_module, fromlist=[f_name])
        im_name = getattr(im_module, f_name)
        if method is None:
            return im_name
        else:
            return getattr(im_name, method)
    except:
        if method is not None:
            f_name = f_name + '.' + method
        print('WARNING: cannot import ' + f_name + ' from ' + f_module + ', check the file TROUBLESHOOTING.md')
        return None


# manager to manage the points features and descriptors
class points_manager:
    def __init__(self, number_features=2000, number_levels=4, scale=1.2,
                 detector_type=DetectorTypes.FAST, descriptor_type=DescriptorTypes.FAST):
        self.detector_type = detector_type
        self.feature_type = None
        self.descriptor_type = descriptor_type
        self.number_features = number_features
        self.number_levels = number_levels
        self.scale = scale

        self.norm_type = None

        self.do_keypoints_size_rescaling = False  # managed below depending on selected features
        self.keypoint_filter_type = KeyPointFilterTypes.SAT  # keypoint-filter type
        self.need_nms = False  # non-maximum suppression  needed
        self.keypoint_nms_filter_type = KeyPointFilterTypes.KDT_NMS  # default keypoint-filter type if NMS is needed
        self.sigma_level0 = kSigmaLevel0

        self.keypoint_filter_type = KeyPointFilterTypes.SAT

        # sigmas for keypoint levels
        self.init_sigma_levels()

        self.need_color_image = False

        # features
        # detector
        self.FAST_create = import_f('cv2', 'FastFeatureDetector_create')
        self.need_nms = False
        if self.detector_type == DetectorTypes.FAST:
            self._detector = self.FAST_create(threshold=20, nonmaxSuppression=True)
            if self.descriptor_type != DescriptorTypes.NONE:
                # self.use_bock_adaptor = True  # override a block adaptor?
                self.use_pyramid_adaptor = self.number_levels > 1  # override a pyramid adaptor?
                self.need_nms = self.number_levels > 1
                self.keypoint_nms_filter_type = KeyPointFilterTypes.OCTREE_NMS
                self.do_keypoints_size_rescaling = True
        else:
            raise ValueError("Unknown feature detector %s" % self.detector_type)

        if self.need_nms:
            self.keypoint_filter_type = self.keypoint_nms_filter_type

        try:
            self.norm_type = FInfo.norm_type[self.descriptor_type]
        except:
            print('You did not set the norm type for: ', self.descriptor_type.name)
            raise ValueError("Unmanaged norm type for feature descriptor %s" % self.descriptor_type.name)

        #  descriptor distance functions
        if self.norm_type == cv2.NORM_HAMMING:
            self.descriptor_distance = hamming_distance
            self.descriptor_distances = hamming_distances
        if self.norm_type == cv2.NORM_L2:
            self.descriptor_distance = l2_distance
            self.descriptor_distances = l2_distances

        print('points manger is created')

    # initialize scale factors, sigmas for each octave level
    def init_sigma_levels(self):
        print('num_levels: ', self.number_levels)
        num_levels = max(kNumLevelsInitSigma, self.number_levels)
        self.inv_scale_factor = 1. / self.scale
        self.scale_factors = np.zeros(num_levels)
        self.level_sigmas2 = np.zeros(num_levels)
        self.level_sigmas = np.zeros(num_levels)
        self.inv_scale_factors = np.zeros(num_levels)
        self.inv_level_sigmas2 = np.zeros(num_levels)
        self.log_scale_factor = math.log(self.scale)

        self.scale_factors[0] = 1.0
        self.level_sigmas2[0] = self.sigma_level0 * self.sigma_level0
        self.level_sigmas[0] = math.sqrt(self.level_sigmas2[0])
        for i in range(1, num_levels):
            self.scale_factors[i] = self.scale_factors[i - 1] * self.scale
            self.level_sigmas2[i] = self.scale_factors[i] * self.scale_factors[i] * self.level_sigmas2[0]
            self.level_sigmas[i] = math.sqrt(self.level_sigmas2[i])
        for i in range(num_levels):
            self.inv_scale_factors[i] = 1.0 / self.scale_factors[i]
            self.inv_level_sigmas2[i] = 1.0 / self.level_sigmas2[i]

    # filter matches by using
    # or SAT (get features with best responses)
    def filter_keypoints(self, type, frame, kps, des=None):
        _name = type.name
        if type == KeyPointFilterTypes.NONE:
            pass
        elif type == KeyPointFilterTypes.SAT:
            if len(kps) > self.number_features:
                keypoints, des = sat_num_features(kps, des, self.number_features)
        else:
            raise ValueError("Unknown match-filter type")
        return keypoints, des, _name

    def keypoints_rescale(self, keypoints):
        # if keypoints are FAST, etc. then rescale their small sizes
        # in order to let descriptors compute an encoded representation with a decent patch size
        scale = 1
        doit = False
        if self.detector_type == DetectorTypes.FAST:
            scale = kFASTKeyPointSizeRescaleFactor
            doit = True

        if doit:
            for keypoint in keypoints:
                keypoint.size *= scale

    # out: kps (array of cv2.KeyPoint)
    def detect(self, frame, mask=None, filter=True):
        if not self.need_color_image and frame.ndim > 2:  # check if we have to convert to gray image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        else:
            # standard detection
            kps = self._detector.detect(frame, mask)
            # filter keypoints
        filter_name = 'NONE'
        if filter:
            kps, _, filter_name = self.filter_keypoints(self.keypoint_filter_type, frame, kps)
            # if keypoints are FAST, etc. give them a decent size in order to properly compute the descriptors
        if self.do_keypoints_size_rescaling:
            self.rescale_keypoint_size(kps)
        return kps

        # compute the descriptors once given the keypoints

    def compute(self, frame, kps, filter=True):
        if not self.need_color_image and frame.ndim > 2:  # check if we have to convert to gray image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kps, des = self._feature_descriptor.compute(frame, kps)  # then, compute descriptors
        # filter keypoints
        filter_name = 'NONE'
        if filter:
            kps, des, filter_name = self.filter_keypoints(self.keypoint_filter_type, frame, kps, des)
        return kps, des

        # detect keypoints and their descriptors

    # out: kps, des
    def detectAndCompute(self, frame, mask=None, filter=True):
        if not self.need_color_image and frame.ndim > 2:  # check if we have to convert to gray image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # standard detectAndCompute
        if self.is_detector_equal_to_descriptor:
            # detector = descriptor => call them together with detectAndCompute() method
            kps, des = self._feature_detector.detectAndCompute(frame, mask)

        else:
            # detector and descriptor are different => call them separately
            # 1. first, detect keypoint locations
            kps = self.detect(frame, mask, filter=False)
            # 2. then, compute descriptors
            kps, des = self._feature_descriptor.compute(frame, kps)

        filter_name = 'NONE'
        if filter:
            kps, des, filter_name = self.filter_keypoints(self.keypoint_filter_type, frame, kps, des)

        return kps, des


def points_manager_factory(number_features=2000, number_levels=4, scale=1.2,
                           detector_type=DetectorTypes.FAST, descriptor_type=DescriptorTypes.FAST):
    return points_manager(number_features, number_levels, scale, detector_type, descriptor_type)
