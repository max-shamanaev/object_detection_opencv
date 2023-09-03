# object_detection_opencv
The goal of this project is to get the basic understanding about object detection using OpenCV library with pretrained neural network.
Implementation of object detection for now relies only on YOLOv8 model hence other models probably won't work correctly.

CUDA and cuDNN were used to drastically improve performance when using detection in real-time or for video captures. 

# Dependencies
Project was tested only with below-mentioned versions of dependencies with OpenCV library being the main one.
In theory, object_detection should correctly run with OpenCV v4 and newer, but there were problems building OpenCV 4.8.0 with CUDA so downgrade was made.

C++ Standard - >=17<br>
CMake - >=3.5
OpenCV - 4.7.0<br>
Neural Network Models - YOLOv8<br>
CUDA (optional) - 11.4<br>
cuDNN (optional) - 8.9.2.26<br>

C++17 used only for std::string_view which can be entirely replaced with normal std::string if older version of C++ is needed.
Adding the filesystem feature is in consideration for possible future improvements so it was decided not to downgrade the version of the standard for now. 

Although not strictly required, CUDA and cuDNN help a lot with performance. 

# Build
1) Clone repository
2) Make sure to meet the dependencies requirements
3) Run build.bat
