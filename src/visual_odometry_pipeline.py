import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from plotting import Plotter

class VisualOdometryPiper:

    def __init__(self, K, data_dir,number_of_images,plot_live):
        self.K = K
        self.FLANN_INDEX_LSH = 6
        self.plot_live = plot_live
        self.images = self.__load_images(data_dir)
        self.images = self.images[:number_of_images]
        self.orb = cv2.ORB_create(3000)
        self.index_params = dict(algorithm=self.FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        self.search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=self.index_params, searchParams=self.search_params)
        self.all_path = []
        self.all_b_pose = []
        self.q_initial = []
        self.q_current = []
        self.plotter = Plotter()
        self.line1 = []

    def __load_images(self, filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    def visualize(self):
        for i,_ in enumerate(tqdm(self.images, unit="images")):

            if i == 0:
                base = np.hstack((np.eye(3), np.zeros((3, 1))))
                base = np.r_[base, [np.array([0, 0, 0, 1])]]
                cur_pose = base

            else:
                ### Get Matches ####
                # Find the keypoints and descriptors with ORB
                kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
                kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
                # Find matches
                matches = self.flann.knnMatch(des1, des2, k=2)

                # Find the matches that do not have distance which is too high
                good_matches = []
                try:
                    for m, n in matches:
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                except ValueError:
                    pass

                # Get the image points from the good matches
                q_last = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                self.q_current = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                self.q_initial = self.q_current

                E, E_mask = cv2.findEssentialMat(q_last, self.q_current, self.K, prob=0.999, threshold=0.45)
                q_last = np.multiply(q_last, E_mask)
                q_last = q_last[q_last[:, 0] != 0]
                self.q_current = np.multiply(self.q_current, E_mask)
                self.q_current = self.q_current[self.q_current[:, 0] != 0]

                # Pose returns transform from 2 to 1 look at slides!!!
                pose = cv2.recoverPose(E, q_last, self.q_current, self.K, distanceThresh=50)

                transf = np.hstack((pose[1], pose[2]))
                transf = np.r_[transf, [np.array([0, 0, 0, 1])]]

                rec_mask = pose[3]
                rec_mask[rec_mask == 255] = 1
                q_last = np.multiply(q_last, rec_mask)
                q_last = q_last[q_last[:, 0] != 0]
                self.q_current = np.multiply(self.q_current, rec_mask)
                self.q_current = self.q_current[self.q_current[:, 0] != 0]

                triangulatedPoints = pose[4]
                triangulatedPoints = triangulatedPoints / triangulatedPoints[3]
                triangulatedPoints = triangulatedPoints.T
                triangulatedPoints = np.multiply(triangulatedPoints, rec_mask)
                triangulatedPoints = triangulatedPoints[triangulatedPoints[:, 0] != 0]

                if len(triangulatedPoints) > 7:
                    cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
                    # This part could be used for SLAM
                    # for r, row in enumerate(triangulatedPoints):
                    #     point = cur_pose@row.T
                    #     triangulatedPoints[r] = point.T
                    #
                    # triangulatedPoints = triangulatedPoints[:,:3].astype('double')
                    # q_last = q_last.astype('double')
                    # self.q_current = self.q_current.astype('double')
                    #
                    # if triangulatedPoints.shape[0] >= 4:
                    #     _, rvec, tvec, _ = cv2.solvePnPRansac(triangulatedPoints, self.q_current, self.K, None)
                    #     R, _ = cv2.Rodrigues(rvec)
                    #     back_pose = np.hstack((R, tvec))
                    #     back_pose = np.r_[back_pose, [np.array([0, 0, 0, 1])]]
                    #     back_pose = np.linalg.inv(back_pose)
                    #
                    #     # Filter out heavy outliers
                    #     if abs(back_pose[0, 3]) + abs(back_pose[1, 3]) + abs(back_pose[2, 3]) < 1000:
                    #         self.all_b_pose.append([back_pose[0, 3], back_pose[1, 3], back_pose[2, 3]])
                    # else:
                    #     print("No")
                else:
                    base = np.hstack((np.eye(3), np.zeros((3, 1))))
                    base = np.r_[base, [np.array([0, 0, 0, 1])]]
                    cur_pose = np.matmul(cur_pose, base)
                    print("No move")



            self.all_path.append([cur_pose[0, 3], cur_pose[1, 3], cur_pose[2, 3]])
            if self.plot_live:
                kk = np.array(self.all_path)
                self.line1 = self.plotter.live_plotter(i,self.images[i],self.q_initial,self.q_current,kk[:,0], kk[:,2],self.line1)

        ## Plot
        if self.plot_live:
            plt.ioff()
            plt.show()
