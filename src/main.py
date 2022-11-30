import cv2
import numpy as np
import initialization as init
import continuous_update as cont_update

def main():
    # Kitti Dataset
    K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                    [0, 7.188560000000e+02, 1.852157000000e+02],
                    [0, 0, 1]])


    init.initialization(K)
    cont_update.continuous_update(K)


if __name__ == "__main__":
    main()