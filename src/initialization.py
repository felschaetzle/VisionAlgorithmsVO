import cv2
import numpy as np
import sys
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


# sys.path.append('data')
# print(sys.path)
def initialization(K: np.array):
    img_0 = cv2.imread('../data/kitti/05/image_0/000000.png', cv2.IMREAD_GRAYSCALE)
    img_1 = cv2.imread('../data/kitti/05/image_0/000005.png', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('../data/kitti/05/image_0/000000.png')

    # img_0 = cv2.imread('../data/ex6_data/0001.jpg', cv2.IMREAD_GRAYSCALE)
    # img_1 = cv2.imread('../data/ex6_data/0002.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('../data/ex6_data/0001.jpg')
    # p1 = np.loadtxt('../data/ex6_data/matches0001.txt')
    # p2 = np.loadtxt('../data/ex6_data/matches0002.txt')
    # p1 = p1.T
    # p2 = p2.T

    # p1 = np.r_[p1, np.ones((1, p1.shape[1]))]
    # p2 = np.r_[p2, np.ones((1, p2.shape[1]))]

    # cv2.imshow('img', img_0)
    # cv2.imshow('img1', img_1)
    # cv2.waitKey(0)

    feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.2,
                       minDistance = 3,
                       blockSize = 5)
    unfilter_key_points_0 = cv2.goodFeaturesToTrack(img_0, mask=None, **feature_params)

    # parameters = dict(winSize=(10, 10),
    # maxLevel=2,
    # criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    # 10, 0.03))

    filterd_key_points_1, st, err = cv2.calcOpticalFlowPyrLK(img_0, img_1, unfilter_key_points_0, None)

    # filterd_features_1 = np.around(filterd_features_1)
    #
    if filterd_key_points_1 is not None:
        key_points_1 = filterd_key_points_1[st==1]
        key_points_0 = unfilter_key_points_0[st==1]


    #key_points_0, key_points_1 = get_matches(img_0, img_1)

    normalized_key_points_0 = key_points_0.copy()
    normalized_key_points_1 = key_points_1.copy()

    normalized_key_points_0[:,0] = (key_points_0[:,0]-K[0,2])/K[0,0]
    normalized_key_points_0[:,1] = (key_points_0[:,1]-K[1,2])/K[1,1]

    normalized_key_points_1[:,0] = (key_points_1[:,0]-K[0,2])/K[0,0]
    normalized_key_points_1[:,1] = (key_points_1[:,1]-K[1,2])/K[1,1]


    E = cv2.findEssentialMat(key_points_0, key_points_1, K, method=cv2.RANSAC, prob=0.999, threshold=0.5)

    bitmask = E[1]

    co0 = key_points_0
    co1 = key_points_1
    key_points_0 = []
    key_points_1 = []
    for i, elm in enumerate(bitmask):
        if elm:
            key_points_0.append(co0[i])
            key_points_1.append(co1[i])

    key_points_0 = np.array(key_points_0)
    key_points_1 = np.array(key_points_1)

    RT_candidates = cv2.decomposeEssentialMat(E[0])
    base = np.hstack((np.eye(3),np.zeros((3,1))))

    # r_candidate_1 = Rotation.from_matrix(R1).as_euler('zxz', degrees=True)
    # r_candidate_2 = Rotation.from_matrix(R2).as_euler('zxz', degrees=True)
    # summ_r_candidate_1 = np.sum(r_candidate_1)
    # summ_r_candidate_2 = np.sum(r_candidate_2)

    # # get R with minimal turning angle
    # if summ_r_candidate_1 > summ_r_candidate_2:
    #     R = R2
    # else:
    #     R = R1
    # # make sure z component of t is positive
    # # TODO: Buggy fix
    # if T[2] < 0:
    #     T = -T
    infront_lens_points = 0
    R = np.zeros((3,3))
    T = np.zeros((3,1))
    triangulated_landmarks = np.zeros((4,1))
    for i in range(2):
        print(i)
        for j in range(-1,2,2):
            print(j)
            frameRT = np.hstack((RT_candidates[i],RT_candidates[2]*j))
            # frameRT = np.hstack((R1, -T))
            triangulated_landmarks_can = cv2.triangulatePoints(K@base, K@frameRT, key_points_0.T, key_points_1.T)
            triangulated_landmarks_can = triangulated_landmarks_can/triangulated_landmarks_can[3]
            p_front = np.sum(triangulated_landmarks_can[2,:]>0)
            T_can = RT_candidates[2]*j
            if p_front > infront_lens_points:
                R = RT_candidates[i]
                T = RT_candidates[2]*j
                triangulated_landmarks = triangulated_landmarks_can
                infront_lens_points = p_front

    pixel_coor = []
    for i in range(triangulated_landmarks.shape[1]):
        landmark = np.array([triangulated_landmarks[0,i], triangulated_landmarks[1,i], triangulated_landmarks[2,i]])
        prod = np.matmul(K,landmark)
        prod = prod/prod[2]
        prod = [(prod[0]/K[0,0]) + K[0,2], (prod[1]/K[0,0]) + K[1,2]]
        pixel_coor.append(prod)
    
    print(triangulated_landmarks)
    for i in range(triangulated_landmarks.shape[1]-1):
        if abs(triangulated_landmarks[0, i]) > 500 or abs(triangulated_landmarks[1, i]) > 500 or abs(triangulated_landmarks[2, i]) > 500:
            triangulated_landmarks = np.delete(triangulated_landmarks, i , axis=1)
    pixel_coor = np.array(pixel_coor)

    # pixel_coor = (pixel_coor/K[1,])*K[1,1]
    # print(pixel_coor)
    # uv = K @ 

    # reprojected_pixel_coordinateds = cv2.projectPoints(triangulated_landmarks[:3], np.eye(3), np.zeros((3,1)), K, None)

    # print(reprojected_pixel_coordinateds)
    # points = zip(triangulated_landmarks[0], triangulated_landmarks[1], triangulated_landmarks[2])
    # print(points[0])
    # for i in points:
    #     print(i)


    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection='3d')

    ax.scatter(triangulated_landmarks[0,:], triangulated_landmarks[1,:], triangulated_landmarks[2,:], marker = 'o')
    ax.scatter([0,0],[0,0],[0,1],marker="x")
    cam2 = np.array([[0],[0],[1]])
    cam2 = R@cam2
    print(R,T)
    ax.scatter([T[0],T[0]+cam2[0]],[T[1],T[1]+cam2[1]],[T[2],T[0]+cam2[2]], marker="x")
    ax = fig.add_subplot(1,3,2)
    ax.imshow(img_0)
    ax.scatter(key_points_0[:,0], key_points_0[:,1], color = 'y', marker='s')
    ax.set_title("Image 1")

    ax = fig.add_subplot(1,3,3)
    ax.imshow(img_1)
    ax.scatter(key_points_1[:,0], key_points_1[:,1], color = 'y', marker='s')
    ax.set_title("Image 2")

    plt.show()






# Iterate over the corners and draw a circle at that location
    # for i in unfilter_features_0:
    #     x,y = i.ravel()
    #     cv2.circle(img,(x,y),5,(0,0,255),-1)


    # cv2.imshow('img', img)
    # cv2.waitKey(0)

def get_matches(img_0, img_1):
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
    orb = cv2.ORB_create(nfeatures=1500, patchSize=31)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)  
    # Find the keypoints and descriptors with ORB
    kp0, des0 = orb.detectAndCompute(img_0, None)
    kp1, des1 = orb.detectAndCompute(img_1, None)
    print(len(kp0), len(kp1))
    # Find matches
    matches = matcher.knnMatch(des0, des1, k=2)
    # matches = brute_force.knnMatch(des0, des1, k=2)

    good = []
    try:
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
    except ValueError:
        pass

    print(len(good))
    draw_params = dict(matchColor = -1, # draw matches in green color
                singlePointColor = None,
                matchesMask = None, # draw only inliers
                flags = 2)

    # Get the image points form the good matches
    q1 = np.float32([kp0[m.queryIdx].pt for m in good])
    q2 = np.float32([kp1[m.trainIdx].pt for m in good])
    return q1, q2


# K = np.array([  [1379.74,   0,          760.35],
#                 [    0,     1382.08,    503.41],
#                 [    0,     0,          1 ]] )

K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                    [0, 7.188560000000e+02, 1.852157000000e+02],
                    [0, 0, 1]])

initialization(K)