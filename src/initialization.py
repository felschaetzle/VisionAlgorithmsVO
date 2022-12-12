import cv2
import numpy as np
import sys
from scipy.spatial.transform import Rotation

# sys.path.append('data')
# print(sys.path)
def initialization(K: np.array):
    img_0 = cv2.imread('data/kitti/05/image_0/000000.png', cv2.IMREAD_GRAYSCALE)
    img_1 = cv2.imread('data/kitti/05/image_0/000010.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('data/kitti/05/image_0/000000.png')


    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
    unfilter_features_0 = cv2.goodFeaturesToTrack(img_0, mask=None, **feature_params)

    # corners = cv2.goodFeaturesToTrack(gray,20,0.01,10)
    # unfilter_features_0 = np.int0(unfilter_features_0)

    # unfilter_features_1 = cv2.goodFeaturesToTrack(img_1, 20, 0.01, 10)
    # corners = cv2.goodFeaturesToTrack(gray,20,0.01,10)
    # unfilter_features_1 = np.int0(unfilter_features_1)
    # print(unfilter_features_0)


    parameters = dict(winSize=(40, 40),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    10, 0.03))

    filterd_features_1, st, err = cv2.calcOpticalFlowPyrLK(img_0, img_1, unfilter_features_0, None, **parameters)


    filterd_features_1 = np.around(filterd_features_1)
    mask = np.zeros_like(img_0)
    frame = img.copy()

    if filterd_features_1 is not None:
        key_points_1 = filterd_features_1[st==1]
        key_points_0 = unfilter_features_0[st==1]
    # draw the tracks


    # print(key_points_0)
    # print(key_points_1)
    # f = cv2.sfm.normalizedEightPointSolver(key_points_0, key_points_1)
    # F = cv2.findFundamentalMat(key_points_0, key_points_1)
    # print(F[0])
    
    # E = K.T * F[0]* K
    # print(E)
    # print(aaa)


    E = cv2.findEssentialMat(key_points_0, key_points_1, K)
    # print(E[0])

    aaa = cv2.recoverPose(E[0], key_points_0, key_points_1, K, distanceThresh=10)
    # print(aaa)


    E = cv2.findEssentialMat(key_points_0, key_points_1, K)

    R1, R2, T = cv2.decomposeEssentialMat(E[0])
    base = np.hstack((np.eye(3),np.zeros((3,1))))

    r_candidate_1 = Rotation.from_matrix(R1).as_euler('zxz', degrees=True)
    r_candidate_2 = Rotation.from_matrix(R2).as_euler('zxz', degrees=True)
    summ_r_candidate_1 = np.sum(r_candidate_1)
    summ_r_candidate_2 = np.sum(r_candidate_2)

    # get R with minimal turning angle
    if summ_r_candidate_1 > summ_r_candidate_2:
        R = R2
    else:
        R = R1
    # make sure z component of t is positive
    if T[2] < 0:
        T = -T

    frameRT = np.hstack((R,T))

    triangulated_landmarks = cv2.triangulatePoints(base, frameRT, key_points_0.T, key_points_1.T)

    # for i, (new, old) in enumerate(zip(key_points_0, key_points_1)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0,0,255), 2)
    #     frame = cv2.circle(frame, (int(a), int(b)), 5, (0,0,255), -1)

    # cv2.imshow('img', frame)
    # cv2.waitKey(0)






# Iterate over the corners and draw a circle at that location
    # for i in unfilter_features_0:
    #     x,y = i.ravel()
    #     cv2.circle(img,(x,y),5,(0,0,255),-1)


    # cv2.imshow('img', img)
    # cv2.waitKey(0)


K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                    [0, 7.188560000000e+02, 1.852157000000e+02],
                    [0, 0, 1]])

initialization(K)