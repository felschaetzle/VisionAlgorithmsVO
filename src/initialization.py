import cv2
import numpy as np

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

    mask = np.zeros_like(img_0)
    frame = img.copy()

    if filterd_features_1 is not None:
        key_points_0 = filterd_features_1[st==1]
        key_points_1 = unfilter_features_0[st==1]
    # draw the tracks

    # f = cv2.sfm.normalizedEightPointSolver(key_points_0, key_points_1)
    F = cv2.findFundamentalMat(key_points_0, key_points_1)
    print(F[0])

    E = K.T * F[0]* K
    print(E)
    [R, T, good] = cv2.recoverPose(E, key_points_0, key_points_1)
    print(R)
    print(T)
    # [R, t, good, mask, triangulatedPoints] = cv.recoverPose(...)
    # [...] = cv.recoverPose(..., 'OptionName', optionValue, ...)

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
