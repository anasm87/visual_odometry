from data_types import *
import sys
import numpy as np
import os


def gt_factory(conf):
    ds_type = conf.ds_type
    path = conf.ds_settings['base_path']
    name = conf.ds_settings['name']
    associations = None

    if ds_type == GTType.SIMPLE:
        name = conf.ds_settings['gt_file']
        return GTSimple(path, name, associations, ds_type)
    else:
        print('without ground truth is not possible to recover the scale')
        return GT(path, name, associations, ds_type)


# base class
class GT(object):
    def __init__(self, path, name, associations=None, type=GTType.NONE):
        self.path = path
        self.name = name
        self.associations = associations
        self.type = type
        self.filename = None
        self.file_assocaitions = None
        self.data = None
        self.scale = 1

    def get_data_line(self, _id):
        return self.data[_id].strip().split()

    def get_pose_absolute_scale(self, _id):
        return 0, 0, 0, 1

    # convert the dataset into simple xyz scale formate
    def convert_to_xyz(self, filename='groundtruth.txt'):
        out_f = open(filename, 'w')
        number_lins = len(self.data)
        for i in range(number_lins):
            x, y, z, scale = self.get_pose_absolute_scale(i)
            if i == 0:
                scale = 1
            out_f.write("%f %f %f %f \n" % (x, y, z, scale))
        out_f.close()


# the ground truth will be read from a simple file where in every line you have xyz,scale format
class GTSimple(GT):
    def __init__(self, path, name, associations=None, type=GTType.NONE):
        super().__init__(path, name, associations, type)
        self.scale = 1
        self.filename = os.path.join(path, name)
        with open(self.filename) as f:
            self.data = f.readlines()
            self.found = True
        if self.data is None:
            sys.exit(
                'ERROR while reading groundtruth file: please, check how you deployed the files and if the code is consistent with this!')

    def get_pose_absolute_scale(self, _id):
        line_ = self.get_data_line(_id - 1)
        x_prev = self.scale * float(line_[0])
        y_prev = self.scale * float(line_[1])
        z_prev = self.scale * float(line_[2])
        line_ = self.get_data_line(_id)
        x = self.scale * float(line_[0])
        y = self.scale * float(line_[1])
        z = self.scale * float(line_[2])
        absolute_scale = np.sqrt(
            (x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))
        return x, y, z, absolute_scale
