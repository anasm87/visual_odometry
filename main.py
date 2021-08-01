# this code is implementing visual slam
# author: Anas Mhana
# purpose : code sample for job application

from conf import Conf
from dataset import ds_factory
from camera import cam_factory
from tracker import tracker_factory
from ground_truth import gt_factory
from visual_odom import VisualOdem


def vo_it():
    print('starting point .....')

    # configuration object
    conf = Conf()

    # dataset factory
    dataset = ds_factory(conf.ds_settings)

    # ground truth object
    gt = gt_factory(conf)

    # camera object factory
    cam = cam_factory(conf)

    # feature tracker object
    tracker = tracker_factory(conf)

    # visual odometry
    visual_odom = VisualOdem(cam, gt, tracker)

    # VO it or do it here
    _id = 0
    while dataset.ok():

        image = dataset.get_image(_id)

        if image is not None:

            visual_odom.track(image, _id)  # main VO function
        _id += 1

    print('The end.....')


def main():
    # lets vo(visual odometry) it
    vo_it()


# main
if __name__ == '__main__':
    main()
