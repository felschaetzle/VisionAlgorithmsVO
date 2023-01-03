import numpy as np
from visual_odometry_pipeline import VisualOdometryPiper

def main():
    K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],[0, 7.188560000000e+02, 1.852157000000e+02],[0, 0, 1]]) # Kitti Dataset
    DATA_DIR = "data/kitti/05/image_0/"
    NUMBER_OF_IMAGES_TO_USE = 2761 # Maximum is 2761
    PLOT_LIVE = True # Enable for live plotting
    COLLECT_DATA = False # Enable for collecting keypoints and candidates in file

    piper = VisualOdometryPiper(K, DATA_DIR, NUMBER_OF_IMAGES_TO_USE, PLOT_LIVE, COLLECT_DATA)
    piper.visualize()

if __name__ == "__main__":
    main()