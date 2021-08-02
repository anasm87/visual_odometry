# Visual odometry project (the purpose is for fun)
python implementation of visual odometry. Lukas kande tracker has been used with FAST features. The design allows to add smothly new features and descriptors.  

# Usage

$ git clone --recursive  https://github.com/anasm87/visual_odometry.git

python3 ./main.py
 
**dependencies**:

* Python 3.6.9
* Numpy (1.18.2)
* OpenCV  
 

# UML design:

| <img src="images/classes_tracker.png"
alt="SLAM"  width="400" height="400" border="1" /> |  <img src="images/classes_camera.png"
alt="VO"  width="200" height="400" border="1" />  |
<img src="images/classes_dataset.png"
alt="Feature Matching"  width="200" height="400" border="1" />  |

|  <img src="images/classes_config.png"
alt="VO"  width="300" height="400" border="1" />  |
<img src="images/classes_visual_odom.png"
alt="Feature Matching"  width="300" height="400" border="1" />  |
 

# datset:
kitti
