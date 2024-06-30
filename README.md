# Two-view Stereo Algorithm
Two-view stereo algorithm for reconstructing 3D scenes from a pair of calibrated stereo images. The code allows you to convert multiple 2D viewpoints captured by a stereo camera into a depth map representing the 3D structure of the scene. The main code is implemented in 'two_view.ipynb' which imports functions from 'two_view_stereo.py'.

# Prerequisites
Prerequisite python libraries are included in 'requirements.txt'. to install them, run:
```sh
pip install -r requirements.txt
```

Note: For the K3D library, which is used to visualize the 3D point clouds in the Jupyter notebook, there are some additional steps after pip installation:
```sh
jupyter nbextension install --py --user k3d
jupyter nbextension enable --py --user k3d
```
# Dataset
This project uses the [Templering dataset](https://vision.middlebury.edu/mview/). The Templering dataset provides multi-view images for evaluating stereo and multi-view stereo reconstruction algorithms. The dataset is utilized to test and validate the two-view stereo algorithm implemented in this project.

# Rectification
The orientation of the images is such that the images are lying horizontally. In each image, the upper left corner pixel has coordinates [u,v] = [0,0]
![image](https://github.com/ShreyaPL/Two-view-Stereo-Algorithm/assets/143954086/e1312f4b-d0fb-48a8-9507-2e326b07427b)
This section of the code addresses the rectifying process. 
Pinhole camera model (P_i = R_i^w * P_w + T_i^w) is used to transform world coordinates to camera frame coordinates.
-  The function compute_right2left_transformation calculates the transformation matrix (P_l = R_lj * P_r + T_lj) and baseline (B) to relate the right camera to the left camera.
-  The function compute_rectification_R computes the rectification rotation matrix (R_rect_i) to transform the left image coordinates into a rectified frame where epipolar lines are horizontal
-  The rectify_2view function completes the rectification process
![image](https://github.com/ShreyaPL/Two-view-Stereo-Algorithm/assets/143954086/c23b8313-9de3-4f35-8000-bd5d937e0eb6)

# Disparity Map
This section addresses computing the disparity map. The function image2patch gets patch buffer for each pixel location using zero-padding. The 'compute_disparity_map' implements three matching metrics (SSD, SAD, and ZNCC) to compare image patches around corresponding pixels in the rectified left and right views. 
By comparing these patches across the entire rectified image, we construct a disparity map that encodes the relative horizontal shift between corresponding pixels.
Disparity Map of one left pixel
![image](https://github.com/ShreyaPL/Two-view-Stereo-Algorithm/assets/143954086/6e904145-f402-4360-b9aa-10db666db772)

# Depth map and Point Cloud Reconstruction
![image](https://github.com/ShreyaPL/Two-view-Stereo-Algorithm/assets/143954086/ba55a0e0-3037-480d-91c2-6772e2fd64c6)
