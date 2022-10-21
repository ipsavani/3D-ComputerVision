# point cloud fusion
given data has 3 images and depth maps of the same scene taken from different angles from left to right.
- detect corners to determine features that will be used to fuse the point clouds.
- used harris corner detector and non-maximum suprresion to cumpute 100 pixels with strongest harris operator response funtion.
- convert detected corners to 3d points using provided depthmaps.
- find putative matches between corners of adjacent images and select top 10 matches with least SAD.
- apply ransac to estimate rigid transformation between 3d points of adjacent images.
- used rigid transformations obtained from above to merge points from 3 images into a single point cloud.
- ![image](https://user-images.githubusercontent.com/35480902/197119979-389abacc-55c2-4b40-b24d-b6922d84065a.png)


# normal estimation
given an image and its depthmap.
- compute 3d points of the image using depthmap, for each point compute the normal of the plane passing through it.
- fix the negative normals and store them into rgb values of the image to get normal mapped image.
