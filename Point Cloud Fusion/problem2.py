
# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Camera Intrinsic
K = np.array((
        [525.0, 0, 319.5],
        [0, 525.0, 239.5],
        [0, 0, 1]))

# Scale Factor
S = 5000

# %%

def compute_3d(dmaprgb):
    coordinates = {}
    dmap = dmaprgb[:,:,0]
    h,w = dmap.shape
    K_inv = np.linalg.inv(K)
    for x in range(h):
        for y in range(w):
            wrld_c = (1/S)*dmap[x,y]*(K_inv@np.array([y,x,1]).T)
            coordinates[(x,y)] = wrld_c
    return coordinates

def get_normal(P):
    # 3xN matrix of 3d points
    P = np.array(P).T
    # Expectation
    exp_A = np.mean(P, axis=1).reshape(-1, 1)
    # Variance
    C = P - exp_A
    # Covariance Matrix
    cov = C @ np.transpose(C)
    # SVD
    U, s, Vt = np.linalg.svd(cov)
    # Normal vector is the eigenvector corrosponding to the smallest eigenValue
    n = U[:,-1]
    d = exp_A.T @ n.reshape(-1,1)
    return n,d[0,0]

def surf_norm(dmap,points,kernel_size):
    # initialize normal map
    nmap = dmap.copy()
    h,w=dmap[:,:,0].shape
    k_half = kernel_size // 2
    zero_v = np.array([0,0,0])
    # loop through each pixel
    for x in range(h):
        for y in range(w):
            # list of existing 3d points
            P = []
            # initialize normal vector
            n = zero_v
            # discard pixel if its a zero vector
            if not np.array_equal(points[(x,y)],zero_v) :
                # if 3d point exists in 7x7 neighborhood of pixel, add to P
                for u in range(x-k_half,x+k_half+1):
                    for v in range(y-k_half,y+k_half+1):
                        if (u>=0 and u<h and v>=0 and v<w and not(np.array_equal(points[(x,y)],zero_v))):
                            P.append(points[(u,v)])
                # check if atleast 3 points exist, otherwise normal vector is (0,0,0)
                if not len(P)<3:
                    # get normal vector for points
                    n,d = get_normal(P)
                    # unify direction of normal vector
                    if d<0:
                        n = -n  
            # normal vector is zero, assign RGB as zero
            if np.array_equal(n,zero_v):
                nmap[x,y,:] = zero_v
            # else assign normalized and scaled normal vector between 0-255
            # note: flipping the RGB values, as cv2 writes images as BGR
            else:
                nmap[x,y,:] = np.flip(((1/2)*(n/np.sqrt(np.sum(n**2)))+np.array([0.5,0.5,0.5]))*255)
    return nmap


# %%
p2img = cv2.imread('./hw4_data/problem2/rgbn.png')
p2dmap = cv2.imread('./hw4_data/problem2/depthn.png')
cors3d = compute_3d(p2dmap)
surface_normal_img = surf_norm(p2dmap,cors3d,7)
cv2.imwrite('surface_normal.png',surface_normal_img)

# %%



