from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

# DLT
#compute affine matrix
def compute_A(X, X_):

    A_list = []
    zero = np.zeros(3)

    for i in range(X.shape[0]):
        w = X_[i, 2]
        y = X_[i, 1]
        x = X_[i, 0]
        XT = X[i]

        A_list.append(
            np.array([[zero, -w * XT, y * XT], [w * XT, zero, -x * XT]]))

    A = np.reshape(np.asarray(A_list), (2*X.shape[0], 9))

    return A

#compute homography
def compute_H(A):

    _, _, Vh = np.linalg.svd(A)
    H = np.reshape(Vh[-1], (3, 3)) 

    return H

#Denormalize image coordinates
def denorm_H(H, T, T_):
    return np.linalg.inv(T_) @ H @ T

#DLT
def DLT(X, X_, h=980, w=500, h_=366, w_=488):

    T = np.linalg.inv(
        np.array([[w+h, 0, w/2], [0, w+h, h/2], [0, 0, 1]]))
    T_ = np.linalg.inv(
        np.array([[w_+h_, 0, w_/2], [0, w_+h_, h_/2], [0, 0, 1]]))

    Xn, Xn_ = np.zeros(X.shape), np.zeros(X.shape)

    for i in range(X.shape[0]):
        Xn[i] = T @ X[i]
        Xn_[i] = T_ @ X_[i]

    A = compute_A(Xn, Xn_)
    H_norm = compute_H(A)
    H = denorm_H(H_norm, T, T_)

    return H

#warping
def cords_warp(i_max, j_max, H, w=1):

    warped_coords = []

    for i in range(i_max):
        for j in range(j_max):
            x = np.array([i, j, w])
            x_ = H @ x
            x_ = x_ / x_[2]
            warped_coords.append(x_)

    Bmap = np.reshape(np.array(warped_coords), (i_max, j_max, 3))[:, :, :-1]

    return Bmap


#interpolation
def biliniar_interpolation_back(Bmap, image):

    warped_image = np.zeros((Bmap.shape[0], Bmap.shape[1], 3))

    for i in range(Bmap.shape[0]):
        for j in range(Bmap.shape[1]):

            x, y = int(Bmap[i, j, 0]), int(Bmap[i, j, 1])

            if x > 0 and y > 0 and x < image.shape[0]-1 and y < image.shape[1]-1:

                d1, d2, d3, d4 = np.sqrt(2) - np.linalg.norm(np.array([x, y])-Bmap[i, j]), np.sqrt(
                    2) - np.linalg.norm(np.array([x, y+1])-Bmap[i, j]), np.sqrt(
                    2) - np.linalg.norm(np.array([x+1, y])-Bmap[i, j]), np.sqrt(
                    2) - np.linalg.norm(np.array([x+1, y+1])-Bmap[i, j])
                warped_image[i, j] = (d1*image[x, y] + d2*image[x, y+1] + d3*image[
                    x+1, y] + d4*image[x+1, y+1])/(d1+d2+d3+d4)

    return warped_image.astype('int32')


im = Image.open(
    "basketball-court.ppm")
# im.show()
img = np.asarray(im)

plt.imshow(img)
plt.show()

z = 1

alpha = 1
#source
X_ = np.array([[54, 248, z], [74, 404, z], [194, 23, z], [280, 280, z]])
w_, h_ = 366, 488
#target
X = np.array([[0, 0, z], [0, 499, z], [979, 0, z], [979, 499, z]])
w, h =  980, 500

H = DLT(X, X_, h, w, h_, w_)
    
Bmap = alpha * np.abs(cords_warp(w, h, H, z))

img_ = biliniar_interpolation_back(Bmap, img)

plt.imshow(img_)
plt.show()

im_ = Image.fromarray(img_.astype('uint8'), 'RGB')
im_.show()

