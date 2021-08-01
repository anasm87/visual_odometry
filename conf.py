import sys

if sys.version_info[0] != 3:
    print("This script requires Python 3")
    exit()

import os
import yaml
import configparser
import numpy as np
from data_types import *

# get the folder location of this file!
local_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# configuration class definition
class Conf(object):
    def __init__(self):

        self.root = local_path
        self.config_file = 'config.ini'
        self.config_parser = configparser.ConfigParser()
        self.cam_params = None
        self.ds_type = None  # type of data set
        self.ds_settings = None  # settings of data set
        self.doc = None
        self.ds_path = None

        self.tracker_type = None
        self.number_features = None
        self.scale = None
        self.detector_type = None
        self.match_ratio_test = None
        self.descriptor_type = None
        self._fps = None
        self.number_levels = None

        # parser to parse the configuration file config.ini
        self.config_parser.read(os.path.join(local_path, self.config_file))
        # call to get the settings required for the datset
        self.get_ds_settings()
        # call to load the camera parameters
        self.get_cam_params()
        # call to load tracker parameters
        self.get_tracker_settings()
        print('configuration object is created.')

    # get the dataset settings
    def get_tracker_settings(self):
        self.number_features = self.config_parser.getint('TRACKER', 'number_features')
        self.scale = self.config_parser.getfloat('TRACKER', 'scale')
        self.match_ratio_test = self.config_parser.getfloat('TRACKER', 'match_ratio_test')
        self.number_levels = self.config_parser.getint('TRACKER','number_levels')

        if self.config_parser['TRACKER']['tracker_type'] == 'LUCASKANDA':
            self.tracker_type = TrackerTypes.LUCASKANADE
        elif self.config_parser['TRACKER']['tracker_type'] == 'NONE':
            self.tracker_type = TrackerTypes.NONE
        else:
            self.tracker_type = None

        if self.config_parser['TRACKER']['descriptor_type'] == 'FAST':
            self.descriptor_type = DescriptorTypes.FAST
        elif self.config_parser['TRACKER']['descriptor_type'] == 'NONE':
            self.descriptor_type = DescriptorTypes.NONE
        else:
            self.descriptor_type = None

        if self.config_parser['TRACKER']['detector_type'] == 'FAST':
            self.detector_type = DetectorTypes.FAST
        elif self.config_parser['TRACKER']['detector_type'] == 'NONE':
            self.detector_type = DetectorTypes.NONE
        else:
            self.detector_type = None





    # get the dataset settings
    def get_ds_settings(self):
        self.ds_type = self.config_parser['DATASET']['type']
        self.doc = os.path.join(local_path, self.config_parser[self.ds_type]['cam_params'])
        self.ds_settings = self.config_parser[self.ds_type]
        self.ds_path = self.ds_settings['base_path']
        self.ds_settings['base_path'] = os.path.join(local_path, self.ds_path)
        if self.config_parser['FOLDER_DATASET']['type'] == 'folder':
            self.ds_type = GTType.SIMPLE
        else:
            self.ds_type = GTType.NONE

    # get the camera params
    def get_cam_params(self):
        self.cam_params = None
        if self.doc is not None:
            with open(self.doc, 'r') as stream:
                try:
                    self.cam_params = yaml.load(stream, Loader=yaml.FullLoader)
                except yaml.YAMLError as exc:
                    print(exc)

    # calibration matrix
    @property
    def K(self):
        if not hasattr(self, '_K'):
            fx = self.cam_settings['Cam.fx']
            cx = self.cam_settings['Cam.cx']
            fy = self.cam_settings['Cam.fy']
            cy = self.cam_settings['Cam.cy']
            self._K = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])
        return self._K

    # inverse of calibration matrix
    @property
    def Kinv(self):
        if not hasattr(self, '_Kinv'):
            fx = self.cam_settings['Cam.fx']
            cx = self.cam_settings['Cam.cx']
            fy = self.cam_settings['Cam.fy']
            cy = self.cam_settings['Cam.cy']
            self._Kinv = np.array([[1 / fx, 0, -cx / fx],
                                   [0, 1 / fy, -cy / fy],
                                   [0, 0, 1]])
        return self._Kinv

    # distortion coefficients
    @property
    def DistCoef(self):
        if not hasattr(self, '_DistCoef'):
            k1 = self.cam_params['Cam.k1']
            k2 = self.cam_params['Cam.k2']
            p1 = self.cam_params['Cam.p1']
            p2 = self.cam_params['Cam.p2']
            k3 = 0
            if 'Camera.k3' in self.cam_params:
                k3 = self.cam_params['Cam.k3']
            self._DistCoef = np.array([k1, k2, p1, p2, k3])
            # if k3 != 0:
            #     self._DistCoef = np.array([k1,k2,p1,p2,k3])
            # else:
            #     self._DistCoef = np.array([k1,k2,p1,p2])
        return self._DistCoef

    # camera width
    @property
    def width(self):
        if not hasattr(self, '_width'):
            self._width = self.cam_params['Cam.width']
        return self._width

    # camera height
    @property
    def height(self):
        if not hasattr(self, '_height'):
            self._height = self.cam_params['Cam.height']
        return self._height

    # camera fps
    @property
    def fps(self):
        if not hasattr(self, '_fps'):
            self._fps = self.cam_params['Cam.fps']
        return self._fps



