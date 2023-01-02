import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from plotting import Plotter

class VisualOdometryPiper:

    def __init__(self, K, data_dir, number_of_images, plot_live):
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
        found_enough_keypoints = True
        for i,_ in enumerate(tqdm(self.images, unit="images")):
            
            if i == 0:
                first_pose = np.hstack((np.eye(3), np.zeros((3, 1))))
                current_pose = np.r_[first_pose, [np.array([0, 0, 0, 1])]]

            else:
                ### Get Matches ####
                # Find the keypoints and descriptors with ORB
                keypoints_last, descriptor_last = self.orb.detectAndCompute(self.images[i - 1], None)
                keypoints_current, descriptor_current = self.orb.detectAndCompute(self.images[i], None)
                # Find matches
                matches = self.flann.knnMatch(descriptor_last, descriptor_current, k=2)

                # Filter matches
                good_matches = []
                try:
                    for m, n in matches:
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                except ValueError:
                    pass

                # Get the image points from the good matches
                q_last = np.float32([keypoints_last[m.queryIdx].pt for m in good_matches])
                self.q_current = np.float32([keypoints_current[m.trainIdx].pt for m in good_matches])
                self.q_initial = self.q_current

                E, E_mask = cv2.findEssentialMat(q_last, self.q_current, self.K, prob=0.999, threshold=0.45)
                q_last = np.multiply(q_last, E_mask)
                q_last = q_last[q_last[:, 0] != 0]
                self.q_current = np.multiply(self.q_current, E_mask)
                self.q_current = self.q_current[self.q_current[:, 0] != 0]

                # Pose returns transform from 2 to 1 look at slides!!!
                # Removes disambiguity of E matrix and calculates the pose
                pose = cv2.recoverPose(E, q_last, self.q_current, self.K, distanceThresh=50)

                transform_matrix = np.hstack((pose[1], pose[2]))
                transform_matrix = np.r_[transform_matrix, [np.array([0, 0, 0, 1])]]

                recover_pose_mask = pose[3]
                recover_pose_mask[recover_pose_mask == 255] = 1
                q_last = np.multiply(q_last, recover_pose_mask)
                q_last = q_last[q_last[:, 0] != 0]
                self.q_current = np.multiply(self.q_current, recover_pose_mask)
                self.q_current = self.q_current[self.q_current[:, 0] != 0]

                triangulated_points = np.multiply(np.transpose(pose[4] / pose[4][3]), recover_pose_mask)
                triangulated_points = triangulated_points[triangulated_points[:, 0] != 0]

                # If at least 7 good landmarks have been triangulated, calculate new pose
                if len(triangulated_points) > 7:
                    current_pose = np.matmul(current_pose, np.linalg.inv(transform_matrix))
                    if(not found_enough_keypoints):
                        print("Found enough keypoints again. Continuing movement.")
                        found_enough_keypoints = True
                    # This part could be used for SLAM
                    #self.__slam(triangulated_points, current_pose)

                # Else, assume we are not moving -> stay and don't update current_pose
                elif(found_enough_keypoints):
                    print("Not enough triangulated points. Assuming no movement.")
                    found_enough_keypoints = False

            self.all_path.append([current_pose[0, 3], current_pose[1, 3], current_pose[2, 3]])
            if self.plot_live:
                updated_path = np.array(self.all_path)
                self.line1 = self.plotter.live_plotter(i,self.images[i], self.q_initial, self.q_current, updated_path[:,0], updated_path[:,2], self.line1)

        ## Plot
        if self.plot_live:
            plt.ioff()
            plt.show()

    def __slam(self, points, pose):
        for r, row in enumerate(points):
            point = pose@row.T
            points[r] = point.T
        
        points = points[:,:3].astype('double')
        q_last = q_last.astype('double')
        self.q_current = self.q_current.astype('double')
        
        if points.shape[0] >= 4:
            _, rvec, tvec, _ = cv2.solvePnPRansac(points, self.q_current, self.K, None)
            R, _ = cv2.Rodrigues(rvec)
            back_pose = np.hstack((R, tvec))
            back_pose = np.r_[back_pose, [np.array([0, 0, 0, 1])]]
            back_pose = np.linalg.inv(back_pose)
        
            # Filter out heavy outliers
            if abs(back_pose[0, 3]) + abs(back_pose[1, 3]) + abs(back_pose[2, 3]) < 1000:
                self.all_b_pose.append([back_pose[0, 3], back_pose[1, 3], back_pose[2, 3]])