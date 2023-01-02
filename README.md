# VisionAlgorithmsVO

This is the repo for the mini-project from the lecture **Vision Algorithms for Mobile Robotics**.
The pipeline has been tested using the KITTI dataset on Windows, macOS and Ubuntu.

## Setup and running of the project

- Create the following directories: `data/kitti/05/image_0/` in the base directory of this project
- Put the images from the KITTI dataset from **camera 0** in the `image_0` folder
- Create an **Anaconda** environment `conda create --name <envName> python=3.9` and activate it `conda activate <envName>` (we used python 3.9.14)
- Install the dependencies from the `requirements.txt` file with `pip install -r requirements.txt`
- Run `main.py` and witness greatness