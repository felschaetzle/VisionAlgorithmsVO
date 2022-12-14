import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
from tqdm import tqdm


class VisualOdometry():
    def __init__(self, data_dir):
        self.K = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        # self.gt_poses = self._load_poses(os.path.join(data_dir,"poses.txt"))
        self.images = self._load_images(data_dir)
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file
        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        # with open(filepath, 'r') as f:
        #     params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        #     P = np.reshape(params, (3, 4))
        #     K = P[0:3, 0:3]
        # K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
        #               [0, 7.188560000000e+02, 1.852157000000e+02],
        #               [0, 0, 1]])

        K = np.array([  [1379.74,   0,          760.35],
                [    0,     1382.08,    503.41],
                [    0,     0,          1 ]] )

        base = np.hstack((np.eye(3), np.zeros((3, 1))))
        return K

    # @staticmethod
    # def _load_poses(filepath):
    #     """
    #     Loads the GT poses
    #     Parameters
    #     ----------
    #     filepath (str): The file path to the poses file
    #     Returns
    #     -------
    #     poses (ndarray): The GT poses
    #     """
    #     poses = []
    #     with open(filepath, 'r') as f:
    #         for line in f.readlines():
    #             T = np.fromstring(line, dtype=np.float64, sep=' ')
    #             T = T.reshape(3, 4)
    #             T = np.vstack((T, [0, 0, 0, 1]))
    #             poses.append(T)
    #     return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images
        Parameters
        ----------
        filepath (str): The file path to image dir
        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object
        Parameters
        ----------
        i (int): The current frame
        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Find the matches there do not have a to high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)

        # img3 = cv2.drawMatches(self.images[i], kp1, self.images[i-1],kp2, good ,None,**draw_params)
        # cv2.imshow("image", img3)
        # cv2.waitKey(2)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix
        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, E_mask = cv2.findEssentialMat(q1, q2, self.K,prob=0.999, threshold=0.45)
        # Benes Edit
        q1 = np.multiply(q1, E_mask)
        q1 = q1[q1[:,0] != 0]
        q2 = np.multiply(q2, E_mask)
        q2 = q2[q2[:,0] != 0]
        pose = cv2.recoverPose(E,q1,q2,self.K,distanceThresh=11)
        # pose returns trans from 2 to 1 look at slides!!!
        # Todo: Somt of  here
        R = pose[1].T
        t = R*pose[2]
        trans = np.hstack((pose[1],pose[2]))
        transfrom_mat = np.r_[trans, [np.array([0,0,0,1])]]
        # only retun used q
        rec_mask = pose[3]
        rec_mask[rec_mask == 255] = 1
        q1 = np.multiply(q1, rec_mask)
        q1 = q1[q1[:,0] != 0]
        q2 = np.multiply(q2, rec_mask)
        q2 = q2[q2[:,0] != 0]
        tri = pose[4]
        tri = tri/tri[3]
        tri = tri.T
        tri = np.multiply(tri, rec_mask)
        tri = tri[tri[:,0] != 0]
        return transfrom_mat, q1,q2, tri

    def triangulate(self, q1, q2, old_pose, curr_pose):

        triangulated_landmarks = cv2.triangulatePoints(self.K @ old_pose, self.K @ curr_pose[:3,:], q1.T, q2.T)
        triangulated_landmarks = triangulated_landmarks / triangulated_landmarks[3]
        return triangulated_landmarks[:3]

def main():
    data_dir = "../data/test/"  # Try KITTI_sequence_2 too
    vo = VisualOdometry(data_dir)

    #play_trip(vo.images)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    all_path = []
    all_tri = []
    point_maske = []
    all_b_pose = []
    q_pres_1 = []
    q_pres_2 = []

    #for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
    for i,images in enumerate(tqdm(vo.images, unit="images")):
        if i < 400:
            if i == 0:
                base = np.hstack((np.eye(3), np.zeros((3, 1))))
                cur_pose = base

            else:
                K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                              [0, 7.188560000000e+02, 1.852157000000e+02],
                              [0, 0, 1]])
                q1, q2 = vo.get_matches(i)
                transf, q1, q2, triangulatedPoints = vo.get_pose(q1, q2)
                old_pose = cur_pose
                cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
                transfrom_mat = np.r_[old_pose, [np.array([0, 0, 0, -0.87])]]
                transfrom_mat_12 = cur_pose #np.r_[cur_pose, [np.array([0, 0, 0, 1])]]
                transfrom_mat_12[:,3] = np.array([-0.12062195, -0.23368521, 0.96480131])
                print(cur_pose)
                for i, row in enumerate(triangulatedPoints):
                    #point = np.array([cur_pose[0,3], cur_pose[1,3], cur_pose[2,3],  1]) + row
                    point = transfrom_mat@row.T
                    triangulatedPoints[i] = point.T
                #triangulatedPoints = vo.triangulate(q1, q2, old_pose, transf)

                for i in range(11): #len(triangulatedPoints)
                    # if abs(triangulatedPoints[i,0]) + abs(triangulatedPoints[i,1]) +abs(triangulatedPoints[i,2]) < 100:
                    all_tri.append([triangulatedPoints[i,0],triangulatedPoints[i,1],triangulatedPoints[i,2]])
                    q_pres_1.append([q1[i,0], q1[i,1]])
                    q_pres_2.append([q2[i,0], q2[i,1]])
                q_pres_1 = np.array(q_pres_1)
                q_pres_2 = np.array(q_pres_2)
                triangulatedPoints.astype('double')
                q1.astype('double')
                q2.astype('double')

                # back_pose = cv2.solvePnPRansac(triangulatedPoints[:6,:3],q2[:6],K, distCoeffs=np.zeros((4,1)), flags=0)
                back_pose = cv2.solvePnP(triangulatedPoints[:6,:3],q2[:6],K, distCoeffs=np.zeros((4,1)), flags=0)
                R = back_pose[1].T
                t = -back_pose[2]
                # b_pose = np.hstack((R, t))
                # b_pose_mat = np.r_[b_pose, [np.array([0, 0, 0, 1])]]
                all_b_pose.append([t[0], t[1], t[2]])
            #gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
            #estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
            all_path.append([cur_pose[0, 3], cur_pose[1, 3], cur_pose[2, 3]])
    all_tri = np.array(all_tri)
    reprojected = []
    for elm in all_tri:
        elm = np.append(elm, 1)
        print(elm)
        su = vo.K@transfrom_mat_12
        prod = su@elm
        prod = prod/prod[2]
        print(prod)
        reprojected.append(prod[:2])
    reprojected = np.array(reprojected)
    # repojected = repojected/repojected[2,:]
    print(reprojected)

    #plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 3, 1, projection='3d')
    # kk = np.array(all_path)
    # ki = np.array(all_tri)
    # # kb = np.array(all_b_pose)
    # ax.scatter(kk[:,0], kk[:,1], kk[:,2],marker='x')
    # ax.scatter(ki[:,0], ki[:,1], ki[:,2],marker='o')
    # # ax.scatter(kb[:, 0], kb[:, 1], kb[:, 2], marker='o')

    # plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    kk = np.array(all_path)
    ki = all_tri
    kb = np.array(all_b_pose)
    ax.scatter(kk[:,0], kk[:,1], kk[:,2],marker='x')
    ax.scatter(ki[:,0], ki[:,1], ki[:,2],marker='o')
    ax.scatter(kb[:, 0], kb[:, 1], kb[:, 2], marker='o')

    ax = fig.add_subplot(1,3,2)
    ax.imshow(vo.images[0])
    ax.scatter(q_pres_1[:,0], q_pres_1[:,1], color = 'y', marker='x')
    ax.scatter(reprojected[:,0], reprojected[:,1], color = 'r', marker='x')

    ax.set_title("Image 1")

    ax = fig.add_subplot(1,3,3)
    ax.imshow(vo.images[1])
    ax.scatter(q_pres_2[:,0], q_pres_2[:,1], color = 'y', marker='x')
    ax.scatter(reprojected[:,0], reprojected[:,1], color = 'r', marker='x')
    ax.set_title("Image 2")

    plt.show()

if __name__ == "__main__":
    main()