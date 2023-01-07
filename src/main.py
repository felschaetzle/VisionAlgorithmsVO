import numpy as np
from visual_odometry_pipeline import VisualOdometryPiper

def main():
    # Parking Dataset
    # K = np.array([[331.37, 0, 320],[0, 369.568, 240],[0, 0, 1]])
    # Kitti Dataset
    K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],[0, 7.188560000000e+02, 1.852157000000e+02],[0, 0, 1]])
    # Malaga Dataset
    # K = np.array([[621.18428, 0, 404.00760], [0, 621.18428, 309.05989], [0, 0, 1]])
    # Make sure that only image files from ONE camera are in this directory!!! NO .TXT FILES
    DATA_DIR = "../data/kitti/05/image_0" 

    PLOT_LIVE = True # Enable for live plotting
    COLLECT_DATA = False # Enable for collecting keypoints and candidates in file
    MAX_ORB_FEATURES = 2500 # Maximum number of orb features to extract from images
    
    piper = VisualOdometryPiper(K, DATA_DIR, PLOT_LIVE, COLLECT_DATA, MAX_ORB_FEATURES)
    piper.visualize()

if __name__ == "__main__":
    main()