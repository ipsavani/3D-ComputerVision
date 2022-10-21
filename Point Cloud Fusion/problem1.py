# %%
import numpy as np
import cv2
from itertools import permutations as pmt
import open3d as o3d

# Ix kernel
D_X = np.array((
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]))

# Iy kernel
D_Y = np.array((
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]))

# Gaussian kernel
GAUSSIAN = np.array((
    [1/273, 4/273, 7/273, 4/273, 1/273],
    [4/273, 16/273, 26/273, 16/273, 4/273],
    [7/273, 26/273, 41/273, 26/273, 7/273],
    [4/273, 16/273, 26/273, 16/273, 4/273],
    [1/273, 4/273, 7/273, 4/273, 1/273]))

# Camera Intrinsic
K = np.array((
        [525.0, 0, 319.5],
        [0, 525.0, 239.5],
        [0, 0, 1]))

# Scale Factor
S = 5000


# %%
# returns top k indices of an nd array
def topk(array,k=100):
    return np.c_[np.unravel_index(np.argpartition(array.ravel(),-k)[-k:],array.shape)]

# function for padding zeros on image boundaries
def pad_zeros(arr,kernel_size):
    k_half = int(kernel_size/2)
    arr = np.pad(arr,((k_half,k_half),(k_half,k_half)),'constant')
    return arr

# rank transform
def rank_transform(image, kernel_size) :
    img_arr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # grayscaling (0-1)
    h,w = img_arr.shape
    # array to store transformed image
    tran_img = np.zeros((h,w), np.uint8)
    # tran_img.shape = h,w
    k_half = kernel_size // 2
    # padding black pixels around both arrays to compute complete rank transform for all pixels
    img_arr = pad_zeros(img_arr,kernel_size)
    tran_img = pad_zeros(tran_img,kernel_size)
    # rank transform
    for y in range(k_half,h+k_half):
        for x in range(k_half,w+k_half):
            rank = 0
            rank = np.sum(np.where(img_arr[y-k_half:y+k_half+1,x-k_half:x+k_half+1]>img_arr[y,x],1,0))
            tran_img[y, x] = rank
    # return de-padded array
    return tran_img[k_half:h+k_half,k_half:w+k_half]

# non max suppresion
def non_max_suppression(img_arr, kernel_size) :
    h,w = img_arr.shape
    mask = np.zeros((h,w), np.uint8)
    k_half = kernel_size // 2
    # make image 480x640 again to get correct (coordinates/indices)
    img_arr = pad_zeros(img_arr,kernel_size+3)
    mask = pad_zeros(mask,kernel_size+3)
    for y in range(k_half,h+k_half):
        for x in range(k_half,w+k_half):
            if not np.max(img_arr[y-k_half:y+k_half+1,x-k_half:x+k_half+1])>img_arr[y,x]:
                mask[y, x] = 1
    return mask*img_arr

# function to convolve image with given kernel
def convolve(img, kernel):
    kernel_size = kernel.shape[0]
    k_half = kernel_size // 2
    h,w = img.shape
    # initialize image to be convolved
    nimg = np.empty((h-kernel_size+1,w-kernel_size+1), dtype=np.float64)
    # Convolve
    for i in range(k_half, h-k_half):
        for j in range(k_half, w-k_half):
            nimg[i-k_half, j-k_half] = np.sum(img[i-k_half:i+k_half+1, j-k_half:j+k_half+1]*kernel)
    return nimg

# detect corners
def detect_corners(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscaling (0-1)

    # Compute the image derivates 𝐼x and 𝐼y for each pixel of the image separately
    Ix = convolve(img, D_X)
    Iy = convolve(img, D_Y)

    # Use these first-order derivatives to compute 𝐼x2,𝐼y2, and 𝐼xy at each pixel.
    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    Ixy = Ix*Iy 

    # Apply Gaussian smoothing to the derivatives using the 5x5 filter
    mm_Ix2 = convolve(Ix2, GAUSSIAN)
    mm_Iy2 = convolve(Iy2, GAUSSIAN)
    mm_Ixy = convolve(Ixy, GAUSSIAN)

    # Compute the Harris operator response function for each pixel.
    res = mm_Ix2*mm_Iy2 - np.square(mm_Ixy) - 0.06*np.square(mm_Ix2 + mm_Iy2)

    # Apply non-maximum suppression on the responses of the Harris operator in 3x3 windows
    candidate_corners = non_max_suppression(res,3)

    # Pick the 100 corners with the strongest response.
    indices = topk(candidate_corners,100)

    return indices

# convert conrners to 3D points
def corners_to_3Dpoints(indices,img):
    # depth
    img = img[:,:,0]
    # K^-1
    K_inv = np.linalg.inv(K)
    # dict to store 3D points for each pixel
    coordinates = {}
    # compute 3d coordinates for each index
    for pt in indices:
        # ignore if d=0
        if not img[pt[0],pt[1]]==0:
            wrld_c = (1/S)*img[pt[0],pt[1]]*(K_inv@np.array([pt[1],pt[0],1]).T)
            coordinates[(pt[0],pt[1])]=wrld_c
    return coordinates

# calculate SAD in 11x11 windows
def calc_sad(img_right, img_left, idx_right, idx_left,kernel_size):
    k_half = kernel_size // 2
    matches = []
    for left in idx_left:
        for right in idx_right:
            sad = 0
            for i in range(-k_half, k_half+1):
                for j in range(-k_half, k_half+1):
                    try:
                        sad = sad + abs(int(img_left[i + left[0],j + left[1]]) - int(img_right[i + right[0],j + right[1]]))
                    except IndexError:
                        continue
            matches.append([sad, (right[0],right[1]), (left[0],left[1])])
    matches.sort(key=lambda x: x[0])
    return np.array(matches,dtype=object)[:10,1:]

# Match corners
def match_corners(imageR,imageL,indicesR,indicesL):
    # compute the rank transform in 5x5 windows.
    rt_imgL = rank_transform(imageL,5)
    rt_imgR = rank_transform(imageR,5)
    # Use SAD in 11x11 windows (on the rank transformed images) for computing distances.
    matches = calc_sad(rt_imgR,rt_imgL,indicesR,indicesL,11)
    # return top 10 matches
    return matches

# Pose estimation
def estimate_pose(matches,cor2,cor1):
    # ts = 0
    # compute permutations for 3 points out all matches
    match_combs = pmt(range(len(matches)),3)
    # apply RANSAC
    for _,idxs in enumerate(match_combs):
        # ts+=1
        P2 = []
        P1 = []
        # initialize 3 points for images as P2,i and P1,i 
        for _,idx in enumerate(idxs):
            P2.append(cor2[(matches[idx][0][0],matches[idx][0][1])])
            P1.append(cor1[(matches[idx][1][0],matches[idx][1][1])])

        # compute vectors
        v11 = P1[0] - P1[1]
        v12 = P1[1] - P1[2]
        v21 = P2[0] - P2[1]
        v22 = P2[1] - P2[2]

        # compute R
        A = np.array([v21,v22,np.cross(v21,v22)]).T
        B = np.array([v11,v12,np.cross(v11,v12)]).T
        try:
            R_prime = A @ np.linalg.inv(B)
        except:
            continue

        u,s,vt = np.linalg.svd(R_prime)
        R = u @ vt

        # compute t
        t = P2[0] - R@P1[0]

        
        ssd = []
        for _,m in enumerate(matches):
            # get transformed 3D coordinates in Image2 coordinate space
            P2_prime = R @ cor1[(m[1][0],m[1][1])] + t.T
            # convert 3D coordinates to 2D pixel coordinates
            p2_prime = K @ P2_prime
            p2_prime = p2_prime/p2_prime[2]
            # compute SSD with actual and calculated coordinates
            ssd.append(int(p2_prime[0]-m[0][1])**2+int(p2_prime[1]-m[0][0])**2)
        # terminate RANSAC if at least 7 points lie below SSD threshold(3 pixel error)
        if np.sum(np.where(np.array(ssd)<=18,1,0)) > 7:
            break
    # print(ts)
    return R,t

# Finis Coronat Opus.
def to_3d(imagergb,dmaprgb,R,t,rigid = True):
    dmap = dmaprgb[:,:,0]
    h,w = dmap.shape
    K_inv = np.linalg.inv(K)
    coors_3d = []
    colors = []
    for x in range(h):
        for y in range(w):
            if not dmap[x,y]==0:
                wrld_c = (1/S)*dmap[x,y]*(K_inv@np.array([y,x,1]).T)
                if rigid:
                    t_wrld_c = R@wrld_c + t
                else:
                    t_wrld_c = wrld_c
                coors_3d.append(t_wrld_c)
                colors.append(np.flip(imagergb[x,y,:]))
    return coors_3d,colors

# %%
# Read Images
img_rgb1 = cv2.imread('./hw4_data/problem1/rgb1.png')
img_rgb2 = cv2.imread('./hw4_data/problem1/rgb2.png')
img_rgb3 = cv2.imread('./hw4_data/problem1/rgb3.png')

dmap1 = cv2.imread('./hw4_data/problem1/depth1.png')
dmap2 = cv2.imread('./hw4_data/problem1/depth2.png')
dmap3 = cv2.imread('./hw4_data/problem1/depth3.png')

# Part 1. Harris Corner Detection.
indices1 = detect_corners(img_rgb1)
indices2 = detect_corners(img_rgb2)
indices3 = detect_corners(img_rgb3)

# Part 2. Corners to 3D points.
coor1 = corners_to_3Dpoints(indices1,dmap1)
coor2 = corners_to_3Dpoints(indices2,dmap2)
coor3 = corners_to_3Dpoints(indices3,dmap3)

# Part 3. Corner Matching.
top_dist_img21 = match_corners(img_rgb2,img_rgb1,list(coor2.keys()),list(coor1.keys()))
top_dist_img32 = match_corners(img_rgb3,img_rgb2,list(coor3.keys()),list(coor2.keys()))

# Part 4. Pose estimation.
R1,t1 = estimate_pose(top_dist_img21,coor2,coor1)
R3,t3 = estimate_pose(np.fliplr(top_dist_img32),coor2,coor3)

# Part 5. Finis Coronat Opus.
pclpoint1,color1 = to_3d(img_rgb1,dmap1,R1,t1,True)
pclpoint3,color3 = to_3d(img_rgb3,dmap3,R3,t3,True)
pclpoint2,color2 = to_3d(img_rgb2,dmap2,R1,t1,False)

pclpoint1.extend(pclpoint3)
color1.extend(color3)
pclpoint1.extend(pclpoint2)
color1.extend(color2)

pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(np.array(pclpoint1))
pcl.colors = o3d.utility.Vector3dVector(np.array(color1).astype(float)/255)
o3d.io.write_point_cloud('mypcl.ply',pcl)


