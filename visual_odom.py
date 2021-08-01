import numpy as np
from data_types import *
import cv2

# feature tracking and camera poe estimation, in this class  VO basics are used. at each step, it estimates the current
# pose with respect to the previous frame. because the estimation is up to scale the ground truth is used to recover the
# currect the scale

useGTScale = True
kRansacThresholdPixels = 0.1
RansacProb = 0.999
useEsenssialEstim = True
RansacThresholdNormalized = 0.0003
AbsoluteScaleThreshold = 0.1

def pose_rt(rotation, translation):
    ret = np.eye(4)
    ret[:3, :3] = rotation
    ret[:3, 3] = translation
    return ret

class VisualOdem(object):
    def __init__(self, cam, gt, tracker):
        self.state = VOstate.NOT_YET_INITIALIZED
        self.cam = cam
        self.current_image = None
        self.previous_image = None

        self.key_points_reference = None
        self.descriptors_reference = None

        self.key_points_current = None
        self.descriptors_current = None

        self.current_rotation = np.eye(3, 3)
        self.current_translation = np.zeros((3, 1))

        self.X, self.Y, self.Z = None, None, None
        self.gt = gt
        self.tracker = tracker
        self.result = None

        self.matched_keypoints_mask = None
        self.history_initiated = True
        self.poses = []  # all the previous poses
        self.estimated_translation = None  # estimated translations
        self.estimated_translation_gt = None  # estimated translations
        self.estimated_tramslations_centered = []  # estimated translation centered with respect to the first one
        self.estimated_translations_gt_centered = []  # ground truth estimated translations centered with to the first
        # one

        self.number_current_matched_points = None
        self.number_inliers = None

    # get the absolute scale form the ground truth (because the scale is amigius)
    def get_absolute_scale(self, _id):
        if self.gt is not None and useGTScale:
            self.X, self.Y, self.Z, scale = self.gt.get_pose_absolute_scale(_id)
            return scale
        else:
            self.X, self.Y, self.Z = 0, 0, 0
            return 1

    # compute the fundamental matrix
    def compute_fundamental_matrix(self, key_points_reference, key_points_current):
        fundamental, mask = cv2.findFundamentalMat(key_points_reference, key_points_current, cv2.FM_RANSAC,
                                                   param1=kRansacThresholdPixels,
                                                   param2=RansacProb)
        if fundamental is None or fundamental.shape == (1, 1):
            # no fundamental matrix found
            raise Exception('No fundamental matrix found')
        elif fundamental.shape[0] > 3:
            # more than one matrix found, just pick the first
            fundamental = fundamental[0:3, 0:3]
        return np.matrix(fundamental), mask

    # remove outliers
    def remove_outliers_using_mask(self, mask):
        if mask is not None:
            num = self.key_points_current.shape[0]
            mask_indxs = [i for i, v in enumerate(mask) if v > 0]
            self.key_points_current = self.key_points_current[mask_indxs]
            self.key_points_reference = self.key_points_reference[mask_indxs]
            if self.descriptors_current is not None:
                self.descriptors_current = self.descriptors_current[mask_indxs]
            if self.descriptors_reference is not None:
                self.descriptors_reference = self.descriptors_reference[mask_indxs]

    # estimate the pose
    def estimate_pose(self, key_points_reference, key_points_current):
        undistorted_key_points_reference = self.cam.undistort_points(key_points_reference)
        undistorted_key_points_current = self.cam.undistort_points(key_points_current)

        self.key_points_reference = self.cam.undistort_points(undistorted_key_points_reference)
        self.key_points_current = self.cam.undistort_points(undistorted_key_points_current)

        if useEsenssialEstim :
            essential, self.matched_keypoints_mask = cv2.findEssentialMat(self.key_points_current, self.key_points_reference, focal=1,
                                                         pp=(0., 0.), method=cv2.RANSAC, prob=RansacProb,
                                                         threshold=RansacThresholdNormalized)
        else:
            # just for the hell of testing fundamental matrix fitting ;-)
            fundamental, self.matched_keypoints_mask = self.computeFundamentalMatrix(undistorted_key_points_current,
                                                                         undistorted_key_points_reference)
            essential = self.cam.K.T @ fundamental @ self.cam.K    # E = K.T * F * K
        _, rotation, translation, mask = cv2.recoverPose(essential, self.key_points_current, self.key_points_reference,
                                                         focal=1, pp=(0., 0.))
        return rotation, translation  # Rrc, trc (with respect to 'ref' frame)


    def frame1_process(self):
        # the first current image detectt
        self.key_points_reference, self.descriptors_reference = self.tracker.detectAndCompute(self.current_image)
        # convert to an array of points
        self.key_points_reference = np.array([x.pt for x in self.key_points_reference], dtype=np.float32)

    def frame_process(self, frame_id):
        # track
        self.result = self.tracker.track(self.previous_image, self.current_image, self.key_points_reference, self.descriptors_reference)

        # estimate pose
        rotaiton, translation = self.estimate_pose(self.result.key_points_ref_matched, self.result.key_points_cur_matched)

        # update keypoints history
        self.key_points_reference = self.result.key_points_ref
        self.key_points_current = self.result.key_points_cur
        self.descriptors_current = self.result.des_cur
        self.number_current_matched_points = self.key_points_reference.shape[0]
        self.number_inliers = np.sum(self.matched_keypoints_mask)
        abs_scale = self.get_absolute_scale(frame_id)
        if abs_scale > AbsoluteScaleThreshold:
            self.current_translation = self.current_translation + abs_scale * self.current_rotation.dot(translation)
            self.current_rotation = self.current_rotation.dot(rotaiton)


        # check: do we have enough features?
        if (self.tracker.tracker_type == TrackerTypes.LUCASKANADE) and (
                self.key_points_reference.shape[0] < self.tracker.number_features):
            self.key_points_current, self.descriptors_current = self.tracker.detectAndCompute(self.current_image)
            # convert to an array
            self.key_points_current = np.array([x.pt for x in self.key_points_current],
                                    dtype=np.float32)  # convert from list of keypoints to an array of points

        self.key_points_reference = self.key_points_current
        self.descriptors_reference = self.descriptors_current
        self.history_update()

    def track(self, image, frame_id):
        if image.ndim > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        assert (image.ndim == 2 and image.shape[0] == (self.cam.height-1) and image.shape[
            1] == (self.cam.width+1)), "Image size mismatched "
        self.current_image = image

        # check states
        if self.state == VOstate.INITIALIZED:
            self.frame_process(frame_id)
        elif self.state == VOstate.NOT_YET_INITIALIZED:
            self.frame1_process()
            self.state = VOstate.INITIALIZED
        self.previous_image = self.current_image



    def history_update(self):
        if (self.history_initiated is True) and (self.X is not None):
            self.estimated_translation = np.array([self.current_translation[0], self.current_translation[1],
                                                   self.current_translation[2]])
            self.estimated_translation_gt = np.array([self.X, self.Y, self.Z])
            self.history_initiated = False
        if (self.estimated_translation is not None) and (self.estimated_translation_gt is not None):
            p = [self.current_translation[0] - self.estimated_translation[0], self.current_translation[1] -
                 self.estimated_translation[1], self.current_translation[2] - self.estimated_translation[2]]

            self.estimated_tramslations_centered.append(p)
            pg = [self.X - self.estimated_translation[0], self.Y - self.estimated_translation[1],
                  self.Z - self.estimated_translation[2]]
            self.estimated_translations_gt_centered.append(pg)
            self.poses.append(pose_rt(self.current_rotation, p))




