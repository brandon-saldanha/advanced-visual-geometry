import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import linalg


def euclidean_trans(theta, tx, ty):
    return np.array([
        [np.cos(theta), -np.sin(theta), tx],
        [np.sin(theta), np.cos(theta), ty],
        [0, 0, 1]
    ])

filename = '/user/bsaldanha2024/AVG lab 1/fig1_6c__.jpg'
img = cv2.imread(filename)

plt.imshow(img, cmap='gray')
pts = np.asarray(plt.ginput(4, timeout=-1))
plt.show()

points = np.array([
    [1053.01298701,  460.10606061],
    [1364.7012987,   641.92424242],
    [1057.34199134,  867.03246753],
    [ 745.65367965,  641.92424242]
])

print('chosen coord: ', points)  # each row is a point

plt.plot(*zip(*points), marker='o', color='r', ls='')
plt.imshow(img)
plt.show()



'''
Affine rectification
'''
print('\n-------- Task 1: Affine rectification --------')
pts_homo = np.concatenate((points, np.ones((4, 1))), axis=1)  # convert chosen pts to homogeneous coordinate
hor_0 = np.cross(pts_homo[0],pts_homo[1]) # the 1st horizontal line
print('@Task 1.1: First line:', hor_0)  
hor_1 = np.cross(pts_homo[2],pts_homo[3]) # the 2nd horizontal line
pt_ideal_0 = np.cross(hor_0,hor_1) #1st ideal point is at intersection of the two unparallelized horizontal lines
pt_ideal_0 /= pt_ideal_0[-1]  # normalize
print('@Task 1.1: first ideal point: ', pt_ideal_0)

ver_0 = np.cross(pts_homo[1],pts_homo[2]) # the 1st vertical line
ver_1 = np.cross(pts_homo[3],pts_homo[0]) # the 2nd vertical line
pt_ideal_1 = np.cross(ver_0,ver_1) # 2nd ideal point is at intersection of the two unparallelized vertical lines
pt_ideal_1 /= pt_ideal_1[-1] # normalize
print('@Task 1.1: second ideal point: ', pt_ideal_1)

l_inf = np.cross(pt_ideal_0,pt_ideal_1); # image of line at inf
l_inf /= l_inf[-1]
print('@Task1.1: line at infinity: ', l_inf)

print('Task 1.2: Construct the projectivity that affinely rectify image')

H = np.array([(1,0,0),(0,1,0),(l_inf)])

print('@Task 1.2: image of line at inf on affinely rectified image: ', (np.linalg.inv(H).T @ l_inf.reshape(-1, 1)).squeeze())

H_E = euclidean_trans(np.deg2rad(0), 50, 250)

affine_img = cv2.warpPerspective(img, H_E @ H, (img.shape[1], img.shape[0]))

affine_pts= (H_E @ H @ pts_homo.T).T # TODO
for i in range(affine_pts.shape[0]):
    affine_pts[i] /= affine_pts[i, -1]

plt.plot(*zip(*affine_pts[:, :-1]), marker='o', color='r', ls='')
plt.imshow(affine_img)
plt.show()


print('-------- End of Task 1 --------\n')
print(affine_pts)

'''
Task 2: Metric rectification
'''
print('\n-------- Task 2: Metric rectification --------')
print('Task 2.1: transform 4 chosen points from projective image to affine image')

def normalize_line(l):
    norm = np.linalg.norm(l[:2])
    return l / norm if norm != 0 else l

# We decided to use corners in "order" A (0), B (1), C (2), D (3)

# At point A: lines AB and AD
l_AB = normalize_line(np.cross(affine_pts[0,:], affine_pts[1,:]))  # AB
l_AD = normalize_line(np.cross(affine_pts[0,:], affine_pts[3,:]))  # AD

# At point C: lines CB and CD
l_CB = normalize_line(np.cross(affine_pts[2,:], affine_pts[1,:]))  # CB
l_CD = normalize_line(np.cross(affine_pts[2,:], affine_pts[3,:]))  # CD

def constraint_row(m, n):
    # constraint: m.T S n = 0 for orthogonal lines m, n
    return np.array([
        m[0]*n[0],
        m[0]*n[1] + m[1]*n[0],
        m[1]*n[1]
    ])

C0 = constraint_row(l_AB, l_AD)  # At A
C1 = constraint_row(l_CB, l_CD)  # At C

C = np.vstack([C0, C1])
print('@Task 2.2: constraint matrix C:\n', C)

print('Task 2.3: Find s by looking for the kernel of C (hint: SVD)')
U, s, Vh = linalg.svd(C)
s = Vh.T[:, -1]
print('@Task 2.3: s = ', s)

mat_S = np.array([
    [s[0], s[1]],
    [s[1], s[2]],
])
print('@Task 2.3: matrix S:\n', mat_S)

eigvals, eigvecs = np.linalg.eig(mat_S)
print("@Task 2.3: eigenvalues:", eigvals)
K = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, a_min=0, a_max=None)))
Kinv = np.linalg.inv(K)

H = np.array([
    (Kinv[0,0], Kinv[0,1], 0),
    (Kinv[1,0], Kinv[1,1], 0),
    (0, 0, 1)
])
print('Task 2.4: Find the projectivity that does metric rectification')

aff_dual_conic = np.array([
    [s[0], s[1], 0],
    [s[1], s[2], 0],
    [0,    0,   0]
])
print('@Task 2.3: image of dual conic on metric rectified image: ', H @ aff_dual_conic @ H.T)

H_E = euclidean_trans(np.deg2rad(0), 30, 80)
H_fin = H_E @ H
eucl_img = cv2.warpPerspective(affine_img, H_fin, (img.shape[1], img.shape[0]))
eucl_pts = (H_fin @ affine_pts.T).T
for i in range(eucl_pts.shape[0]):
    eucl_pts[i] /= eucl_pts[i, -1]

plt.plot(*zip(*eucl_pts[:, :-1]), marker='o', color='r', ls='')
plt.imshow(eucl_img)
plt.show()




