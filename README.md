# VisionAlgorithmsVO
This is the repo for the mini-project from the lecture **Vision Algorithms for Mobile Robotics**.

## Specifications on hardware and dataset
The pipeline has been tested using the KITTI dataset on Windows, macOS and Ubuntu. The screen cast was recorded on Ubuntu 22.04 using 2 threads of an Intel i5 @ 2.3GHz on a machine with 8GB of RAM.

## Setup and how to run the project
- Create the following directories: `data/kitti/05/image_0/` in the base directory of this project and put the images from the KITTI dataset from *camera 0* in the `image_0` folder.
- **Alternatively** change the file path in the `main.py` file to your desired location.  

- Create an *Anaconda* environment `conda create --name <envName> python=3.9` and activate it `conda activate <envName>` (we used python 3.9.14). Install the dependencies from the `requirements.txt` file with `pip install -r requirements.txt`.
- **Alternatively** simply run `conda env create -n <envName> --file environment.yml`.  

- Run `main.py` and witness greatness.

## Screencasts
The screencasts are uploaded to YouTube
- [KITTI](https://youtu.be/LzPxz6JBkss)
- [Parking](https://youtu.be/Y9UnjBag4RE)
- [Malaga](https://youtu.be/ejYT5tZ5rbc)

## Malaga and parking data sets
- Create new folder under data

- Change the directory variable in the `main.py` file accordingly

- Make sure you delete the *K.txt* file out of the images folder from the parking data set

- Make sure you only use either the left or the right image in the malaga data set e.g. using `mv *_right.jpg RightCamera/`