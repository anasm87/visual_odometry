from data_types import *
import cv2
from points_manager import points_manager_factory

kMinNumFeatureDefault = 2000
kLkPyrOpticFlowNumLevelsMin = 3


#   feature tracker base class
class Tracker(object):
    def __init__(self, number_features=kMinNumFeatureDefault,
                 number_levels=1,  # number of pyramid levels for detector and descriptor
                 scale=1.2,  # detection scale   (if it can be set, otherwise it is automatically computed)
                 detector_type=DetectorTypes.FAST,
                 descriptor_type=DescriptorTypes.FAST,
                 match_ratio_test=0.7,
                 tracker_type=TrackerTypes.LUCASKANADE):
        self.detector_type = detector_type
        self.descriptor_type = descriptor_type
        self.tracker_type = tracker_type

        self.points_manager = None  # descriptor and detector
        self.features_matcher = None  # matching descriptors  methods based on BF, FLANN, etc.

    @property
    def number_features(self):
        return self.points_manager.number_features

    @property
    def number_levels(self):
        return self.points_manager.number_levels

    @property
    def scale(self):
        return self.points_manager.scale

    @property
    def norm_type(self):
        return self.points_manager.norm_type

    @property
    def descriptor_distance(self):
        return self.points_manager.descriptor_distance

    @property
    def descriptor_distances(self):
        return self.points_manager.descriptor_distances

        # out: key_points and descriptors

    def detectAndCompute(self, frame, mask):
        return None, None

        # out: TrackingResult()

    def track(self, image_ref, image_cur, key_points_ref, des_ref):
        return TrackingResult()


# Lucas-Kanade Tracker:
# it uses raw pixel patches as "descriptors" and track/"match" by using Lucas Kanade pyr optical tracker
class LukaskanadeTracker(Tracker):
    def __init__(self, number_features=kMinNumFeatureDefault,
                 number_levels=3,  # n pyramid levels for detector
                 scale=1.2,
                 # scale factor for detection (if it can be set, otherwise it is automatically computed)
                 detector_type=DetectorTypes.FAST,
                 descriptor_type=DescriptorTypes.NONE,
                 match_ratio_test=0.7,
                 tracker_type=TrackerTypes.LUCASKANADE):
        super().__init__(number_features=number_features,
                         number_levels=number_levels,
                         scale=scale,
                         detector_type=detector_type,
                         descriptor_type=descriptor_type,
                         tracker_type=tracker_type)

        self.points_manager = points_manager_factory(number_features=number_features,
                                                     number_levels=number_levels,
                                                     scale=scale,
                                                     detector_type=detector_type,
                                                     descriptor_type=descriptor_type)

        optic_flow_num_levels = max(kLkPyrOpticFlowNumLevelsMin, number_levels)
        print('Lukass and kanade Tracker: number of levels on pyramids optical flow: ', optic_flow_num_levels)
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=optic_flow_num_levels,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        print('Lukas Kanade Tracker is created')

    def detectAndCompute(self, frame, mask=None):
        return self.points_manager.detect(frame, mask), None

    def track(self, image_ref, image_cur, key_points_ref, des_ref=None):
        key_points_cur, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, key_points_ref, None,
                                                           **self.lk_params)
        st = st.reshape(st.shape[0])
        res = TrackingResult()
        # res.indexes_ref = (st == 1)
        res.indexes_ref = [i for i, v in enumerate(st) if v == 1]
        res.indexes_cur = res.indexes_ref.copy()
        res.key_points_ref_matched = key_points_ref[res.indexes_ref]
        res.key_points_cur_matched = key_points_cur[res.indexes_cur]
        res.key_points_ref = res.key_points_ref_matched  # with Lukas Kande we follow feature trails hence we can forget
        # unmatched features
        res.key_points_cur = res.key_points_cur_matched
        res.des_cur = None
        return res


class TrackingResult(object):
    def __init__(self):
        self.key_points_ref = None  # all reference key_points Nx2
        self.key_points_cur = None  # all current key_points   Nx2
        self.des_cur = None  # current descriptors   NxD
        self.indexes_ref = None  # matches indexes in reference key_points so,   key_points_ref_matched =
        # key_points_ref[indexes_ref]
        self.indexes_cur = None  # matches indexes in current key_points so,   key_points_cur_matched =
        # key_points_cur[indexes_cur]
        self.key_points_ref_matched = None  # reference matched key_points,
        self.key_points_cur_matched = None  # current matched key_points,


# create new trackers ... tracker type would be set in the config file
def tracker_factory(conf):
    if conf.tracker_type == TrackerTypes.LUCASKANADE:
        return LukaskanadeTracker(number_features=conf.number_features, number_levels=conf.number_levels,
                                  scale=conf.scale,
                                  detector_type=conf.detector_type, descriptor_type=conf.descriptor_type,
                                  match_ratio_test=conf.match_ratio_test, tracker_type=conf.tracker_type)
    else:
        return None
