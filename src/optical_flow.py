import numpy as np
import cv2
import matplotlib.pyplot as plt


def initialization(K: np.array):
    img_0 = cv2.imread('../data/kitti/05/image_0/000000.png', cv2.IMREAD_GRAYSCALE)
    img_1 = cv2.imread('../data/kitti/05/image_0/000005.png', cv2.IMREAD_GRAYSCALE)
    # img = img_0.copy()
    
    # img_0 = cv2.imread('../data/ex6_data/0001.jpg', cv2.IMREAD_GRAYSCALE)
    # img_1 = cv2.imread('../data/ex6_data/0002.jpg', cv2.IMREAD_GRAYSCALE)

    
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Create random colors
    color = (0,0,255)

    # Take first frame and find corners in it
    p0 = cv2.goodFeaturesToTrack(img_0, mask=None, **feature_params)

    p1, st, err = cv2.calcOpticalFlowPyrLK(img_0, img_1, p0, None, **lk_params)

    # Select good points
    q_last = p1[st == 1]
    q0 = p0[st == 1]

    E, E_mask = cv2.findEssentialMat(q0, q_last, K)


    q0 = np.multiply(q0, E_mask)
    q0 = q0[q0[:,0] != 0]
    q_last = np.multiply(q_last, E_mask)
    q_last = q_last[q_last[:,0] != 0]
    pose = cv2.recoverPose(E,q0,q_last, K,distanceThresh=100)


    trans = np.hstack((pose[1],pose[2]))
    transfrom_mat = np.r_[trans, [np.array([0,0,0,1])]]
    # only retun used q
    rec_mask = pose[3]
    rec_mask[rec_mask == 255] = 1
    q0 = np.multiply(q0, rec_mask)
    q0 = q0[q0[:,0] != 0]
    q_last = np.multiply(q_last, rec_mask)
    q_last = q_last[q_last[:,0] != 0]
    tri = pose[4]
    tri = tri/tri[3]
    tri = tri.T
    tri = np.multiply(tri, rec_mask)
    tri = tri[tri[:,0] != 0]

    # K = np.r_[K, [np.array([0, 0, 0, 1])]]
    trans = np.hstack((pose[1],pose[2]))
    transfrom_mat = np.r_[trans, [np.array([0,0,0,1])]]
    transfrom_mat12 = transfrom_mat[:3,:]

    reprojected0 = []
    for elm in tri:
        elm = elm[:3]
        prod = K@elm
        prod = prod/prod[2]
        # print(prod)
        reprojected0.append(prod[:2])
    reprojected0 = np.array(reprojected0)

    reprojected1 = []
    for elm in tri:
        # elm = elm[:3]
        # elm = np.append(elm, 0)
        # print(elm)

        su = K@transfrom_mat12
        prod = su@elm
        prod = prod/prod[2]
        # print(prod)
        reprojected1.append(prod[:2])
    reprojected1 = np.array(reprojected1)



    # mask = np.zeros_like(img_0)
    # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     img = cv2.line(img, (int(a), int(b)), (int(c), int(d)), color, 2)
    #     img = cv2.circle(img, (int(a), int(b)), 5, color, -1)
 
    # # img = cv2.add(frame, mask)
    # # Display the demo
    # cv2.imshow("frame", img)
    # k = cv2.waitKey(0) & 0xFF


    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    # kk = np.array(all_path)
    # ki = all_tri
    # kb = np.array(all_b_pose)
    ki = tri 
    # ax.scatter(kk[:,0], kk[:,1], kk[:,2],marker='x')
    ax.scatter(ki[:,0], ki[:,1], ki[:,2],marker='o')
    # ax.scatter(kb[:, 0], kb[:, 1], kb[:, 2], marker='o')

    ax = fig.add_subplot(1,3,2)
    ax.imshow(img_0)
    ax.scatter(q0[:,0], q0[:,1], color = 'y', marker='x')
    ax.scatter(reprojected0[:,0], reprojected0[:,1], color = 'r', marker='x')

    ax.set_title("Image 1")

    ax = fig.add_subplot(1,3,3)
    ax.imshow(img_1)
    ax.scatter(q_last[:,0], q_last[:,1], color = 'y', marker='x')
    ax.scatter(reprojected1[:,0], reprojected1[:,1], color = 'r', marker='x')
    ax.set_title("Image 2")

    plt.show()


K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                    [0, 7.188560000000e+02, 1.852157000000e+02],
                    [0, 0, 1]])

initialization(K)