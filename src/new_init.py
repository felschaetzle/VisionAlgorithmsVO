import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from plotting import Plotter

def main():
    data_dir = "data/kitti/05/image_0/"  # Try KITTI_sequence_2 too
    plot_live = False # enable for live plotting
    K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02], [0, 7.188560000000e+02, 1.852157000000e+02], [0, 0, 1]])
    images = load_images(data_dir)
    orb = cv2.ORB_create(3000)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    all_path = []
    all_tri = []
    all_b_pose = []
    images = images[:10]
    plotter = Plotter()
    line1 = []
    for i,_ in enumerate(tqdm(images, unit="images")):
        if i == 0:
            base = np.hstack((np.eye(3), np.zeros((3, 1))))
            base = np.r_[base, [np.array([0, 0, 0, 1])]]
            cur_pose = base

        else:
            ### Get Matches ####
            # Find the keypoints and descriptors with ORB

            kp1, des1 = orb.detectAndCompute(images[i - 1], None)
            kp2, des2 = orb.detectAndCompute(images[i], None)
            # Find matches
            matches = flann.knnMatch(des1, des2, k=2)

            # Find the matches there do not have a to high distance
            good = []
            try:
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)
            except ValueError:
                pass

            # Visualize matches
            # draw_params = dict(matchColor=-1, singlePointColor=None, matchesMask=None, flags=2)
            # img3 = cv2.drawMatches(images[i], kp1, images[i-1],kp2, good ,None,**draw_params)
            # cv2.imshow("image", img3)
            # cv2.waitKey(2)

            # Get the image points form the good matches
            q_last = np.float32([kp1[m.queryIdx].pt for m in good])
            q_current = np.float32([kp2[m.trainIdx].pt for m in good])

            E, E_mask = cv2.findEssentialMat(q_last, q_current, K, prob=0.999, threshold=0.45)
            q_last = np.multiply(q_last, E_mask)
            q_last = q_last[q_last[:, 0] != 0]
            q_current = np.multiply(q_current, E_mask)
            q_current = q_current[q_current[:, 0] != 0]

            # pose returns trans from 2 to 1 look at slides!!!
            pose = cv2.recoverPose(E, q_last, q_current, K, distanceThresh=50)

            transf = np.hstack((pose[1], pose[2]))
            transf = np.r_[transf, [np.array([0, 0, 0, 1])]]

            rec_mask = pose[3]
            rec_mask[rec_mask == 255] = 1
            q_last = np.multiply(q_last, rec_mask)
            q_last = q_last[q_last[:, 0] != 0]
            q_current = np.multiply(q_current, rec_mask)
            q_current = q_current[q_current[:, 0] != 0]

            triangulatedPoints = pose[4]
            triangulatedPoints = triangulatedPoints / triangulatedPoints[3]
            triangulatedPoints = triangulatedPoints.T
            triangulatedPoints = np.multiply(triangulatedPoints, rec_mask)
            triangulatedPoints = triangulatedPoints[triangulatedPoints[:, 0] != 0]

            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))

            for r, row in enumerate(triangulatedPoints):
                point = cur_pose@row.T
                triangulatedPoints[r] = point.T

            triangulatedPoints = triangulatedPoints[:,:3].astype('double')
            q_last = q_last.astype('double')
            q_current = q_current.astype('double')

            if triangulatedPoints.shape[0] >= 4:
                _, rvec, tvec, _ = cv2.solvePnPRansac(triangulatedPoints, q_current, K, None)
                R, _ = cv2.Rodrigues(rvec)
                back_pose = np.hstack((R, tvec))
                back_pose = np.r_[back_pose, [np.array([0, 0, 0, 1])]]
                back_pose = np.linalg.inv(back_pose)

                # Filter out heavy outliers# Todo: Make it better!
                if abs(back_pose[0, 3]) + abs(back_pose[1, 3]) + abs(back_pose[2, 3]) < 1000:
                    all_b_pose.append([back_pose[0, 3], back_pose[1, 3], back_pose[2, 3]])
            else:
                print("No")
        
        all_path.append([cur_pose[0, 3], cur_pose[1, 3], cur_pose[2, 3]])
        if plot_live:
            kk = np.array(all_path)
            line1 = plotter.live_plotter(i,images[i],kk[:,0], kk[:,1],line1)


    ## Plot that stuff
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 3, 1, projection='3d')
    if plot_live:
        plt.ioff()
        plt.show()
    # kk = np.array(all_path)
    # ki = np.array(all_tri)
    # kb = np.array(all_b_pose)
    # ax.scatter(kk[:,0], kk[:,1], kk[:,2],marker='x')
    # #ax.scatter(ki[:,0], ki[:,1], ki[:,2],marker='o')
    # ax.scatter(kb[:, 0], kb[:, 1], kb[:, 2], marker='o')

    # plt.show()

def load_images(filepath):
    image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
    return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

if __name__ == "__main__":
    main()



 # def triangulate(self, q_last, q_current, old_pose, curr_pose):
 #
 #        triangulated_landmarks = cv2.triangulatePoints(K @ old_pose, K @ curr_pose[:3,:], q_last.T, q_current.T)
 #        triangulated_landmarks = triangulated_landmarks / triangulated_landmarks[3]
 #        return triangulated_landmarks[:3]