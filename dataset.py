from enum import Enum
import cv2
import glob
import os


from data_types import *


# base  dataset class

class Dataset(object):
    def __init__(self, path, name, fps=None, associations=None, type1=DatasetType.NONE):
        self.path = path
        self.name = name
        self.type = type1
        self.is_ok = True
        self.fps = fps
        self.associations = associations
        if fps is not None:
            self.Ts = 1. / fps
        else:
            self.Ts = None

        self.timestamps = None
        self._timestamp = None  # current timestamp if available [s]
        self._next_timestamp = None  # next timestamp if available otherwise an estimate [s]

    def ok(self):
        return self.is_ok

    def get_image(self, frame_id):
        return None

    def get_image1(self, frame_id):
        return None

    def get_depth(self, frame_id):
        return None

    def get_image_color(self, frame_id):
        try:
            img = self.getImage(frame_id)
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                return img
        except:
            img = None
            # raise IOError('Cannot open dataset: ', self.name, ', path: ', self.path)
            print('Cannot load dataset: ', self.name, ', path: ', self.path)
            return img

    def get_time_stamp(self):
        return self._timestamp

    def get_next_time_stamp(self):
        return self._next_timestamp


# dataset from folder class which inherits from base class

class DatasetFromFolder(Dataset):
    def __init__(self, path, name, fps=None, associations=None, type1=DatasetType.FOLDER):
        super().__init__(path, name, fps, associations, type1)
        if fps is None:
            fps = 10  # default value
        self.fps = fps
        self.Ts = 1. / self.fps
        self.skip = 1
        self.listing = []
        self.maxlen = 1000000

        # getting the paths of all the images in the folder and sort them
        self.listing = glob.glob(path + '/' + self.name)
        self.listing.sort()
        self.listing = self.listing[::self.skip]

        self.maxlen = len(self.listing)
        self.i = 0
        if self.maxlen == 0:
            raise IOError('No images were found in folder: ', path)
        self._timestamp = 0.
        print('dataset from folder object is created.')

    def get_image(self, frame_id):
        if self.i == self.maxlen:
            return None, False
        image_file = self.listing[self.i]
        img = cv2.imread(image_file)
        self._timestamp += self.Ts
        self._next_timestamp = self._timestamp + self.Ts
        if img is None:
            raise IOError('error reading file: ', image_file)
            # Increment internal counter.
        self.i = self.i + 1
        return img


# create datasets objects based on the type in the settings
def ds_factory(settings):
    # local variables
    associations = None
    path = None

    type1 = settings['type']
    name = settings['name']

    path = settings['base_path']
    path = os.path.expanduser(path)

    dataset = None
    if type1 == 'folder':
        if 'fps' in settings:
            fps = int(settings['fps'])
        dataset = DatasetFromFolder(path, name, fps, associations, DatasetType.FOLDER)

    return dataset
