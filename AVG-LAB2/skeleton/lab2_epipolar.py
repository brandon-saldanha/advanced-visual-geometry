import numpy as np
import cv2


# part 1


def normalize_transformation(points: np.ndarray) -> np.ndarray:
    """
    Compute a similarity transformation matrix that translate the points such that
    their center is at the origin & the avg distance from the origin is sqrt(2)
    :param points: <float: num_points, 2> set of key points on an image
    :return: (sim_trans <float, 3, 3>)
    """
    center = np.mean(points, axis=0)  # TODO: find center of the set of points by computing mean of x & y
    shifted = points - center
    dist = np.sqrt(np.sum(shifted**2, axis=1))  # TODO: matrix of distance from every point to the origin, shape: <num_points, 1>
    mean_dist = np.mean(dist)
    s = np.sqrt(2) / mean_dist if mean_dist > 1e-12 else 1.0  # TODO: scale factor the similarity transformation = sqrt(2) / (mean of dist)
    sim_trans = np.array([
        [s,     0,      -s * center[0]],
        [0,     s,      -s * center[1]],
        [0,     0,      1]
    ])
    return sim_trans


def homogenize(points: np.ndarray) -> np.ndarray:
    """
    Convert points to homogeneous coordinate
    :param points: <float: num_points, num_dim>
    :return: <float: num_points, 3>
    """
    return np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)


# read image & put them in grayscale
img1 = cv2.imread('./img/chapel00.png', 0)  # queryImage
img2 = cv2.imread('./img/chapel01.png', 0)  # trainImage

# detect kpts & compute descriptor
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# match kpts
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# organize key points into matrix, each row is a point
query_kpts = np.array([kp1[m.queryIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>
train_kpts = np.array([kp2[m.trainIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>

# normalize kpts
T_query = normalize_transformation(query_kpts)  # get the similarity transformation for normalizing query kpts
normalized_query_kpts = np.array([])  # TODO: apply T_query to query_kpts to normalize them

T_train = normalize_transformation(train_kpts)  # get the similarity transformation for normalizing train kpts
normalized_train_kpts = np.array([])  # TODO: apply T_train to train_kpts to normalize them

query_h = homogenize(query_kpts)   # (N,3)
train_h = homogenize(train_kpts)   # (N,3)

normalized_query_h = (T_query @ query_h.T).T
normalized_train_h = (T_train @ train_h.T).T

normalized_query_kpts = normalized_query_h[:, :2] / normalized_query_h[:, 2:3]
normalized_train_kpts = normalized_train_h[:, :2] / normalized_train_h[:, 2:3]

# construct homogeneous linear equation to find fundamental matrix
x  = normalized_query_kpts[:, 0]  # TODO: construct A according to Eq.(3) in lab subject
y  = normalized_query_kpts[:, 1]
xp = normalized_train_kpts[:, 0]
yp = normalized_train_kpts[:, 1]

A = np.stack([
    xp*x, xp*y, xp,
    yp*x, yp*y, yp,
    x,    y,    np.ones_like(x)
], axis=1)

print("A shape:", A.shape)   # must be (N, 9)

# TODO: find vector f by solving A f = 0 using SVD
# hint: perform SVD of A using np.linalg.svd to get u, s, vh (vh is the transpose of v)
# hint: f is the last column of v
u, svals, vh = np.linalg.svd(A)
f = vh[-1, :]                 # last row of V^T  # TODO: find f

# arrange f into 3x3 matrix to get fundamental matrix F
F = f.reshape(3, 3)
print('rank F: ', np.linalg.matrix_rank(F))  # should be = 3

# TODO: force F to have rank 2
# hint: perform SVD of F using np.linalg.svd to get u, s, vh
# hint: set the smallest singular value of F to 0
# hint: reconstruct F from u, new_s, vh
uF, sF, vhF = np.linalg.svd(F)
sF[-1] = 0.0
F = uF @ np.diag(sF) @ vhF
assert np.linalg.matrix_rank(F) == 2, 'Fundamental matrix must have rank 2'

print("rank F after constraint:", np.linalg.matrix_rank(F))

# TODO: de-normlaize F
# hint: last line of Algorithme 1 in the lab subject
F_gt = np.loadtxt('chapel.00.01.F')
F = T_train.T @ F @ T_query

# NOTE: Fundamental matrix is defined up to a scale, so normalize before comparison
F = F / np.linalg.norm(F)
F_gt = F_gt / np.linalg.norm(F_gt)

print(F - F_gt)

q = homogenize(query_kpts)
t = homogenize(train_kpts)

errs = np.sum(t * ((F @ q.T).T), axis=1)   # x'^T F x for each match

print("Mean |x'^T F x|:", np.mean(np.abs(errs)))
print("Median |x'^T F x|:", np.median(np.abs(errs)))

F_ransac, mask = cv2.findFundamentalMat(query_kpts, train_kpts, cv2.FM_RANSAC, 1.0, 0.99)

F_ransac = F_ransac / np.linalg.norm(F_ransac)

inliers = mask.ravel().astype(bool)
print("Inliers:", np.sum(inliers), "/", len(inliers))

errs_r = np.sum(homogenize(train_kpts[inliers]) * ((F_ransac @ homogenize(query_kpts[inliers]).T).T), axis=1)
print("Mean |x'^T F x| (RANSAC inliers):", np.mean(np.abs(errs_r)))
print("Median |x'^T F x| (RANSAC inliers):", np.median(np.abs(errs_r)))

import matplotlib.pyplot as plt

# pick a few inlier correspondences
idx = np.where(inliers)[0]
sel = idx[:5]

F_use = F_ransac  # use RANSAC F

img1_c = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2_c = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

h2, w2 = img2.shape

for i in sel:
    x1, y1 = query_kpts[i]
    pt1 = np.array([x1, y1, 1.0])

    l2 = F_use @ pt1   # epipolar line in image 2: a x + b y + c = 0
    a, b, c = l2

    # compute 2 points on the line for drawing (x=0 and x=w2-1)
    y0 = int((-c - a * 0) / b)
    y1p = int((-c - a * (w2 - 1)) / b)

    cv2.circle(img1_c, (int(x1), int(y1)), 5, (0, 255, 0), -1)
    cv2.line(img2_c, (0, y0), (w2 - 1, y1p), (0, 255, 0), 2)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img1_c, cv2.COLOR_BGR2RGB)); plt.title("Image 1: selected inlier points"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(img2_c, cv2.COLOR_BGR2RGB)); plt.title("Image 2: epipolar lines (RANSAC F)"); plt.axis("off")
plt.show()