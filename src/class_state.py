class state:
    def __init__(self, P, X, C, F, T):
        self.P = P # keypoints
        self.X = X # 3D landmarks, corresponding to the keypoints
        self.C = C # set of candidate keypoints
        self.F = F # set of first observation of each candidate keypoint
        self.T = T # camera pose at the first observation of each candiated keypoint