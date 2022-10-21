# Voxel based Reconstruction

- DATA folder contains images, silhouettes and projection matrices for a single time instant of a dynamic scene.
- the code obtains a voxel-based reconstruction of the scene using the provided silhouettes as input.
- part 1:
  - created resolution space of 100 million voxels in a grid of size 4*5*2.5 meters.
  - Used camera projection matrix to find image coordinates for each voxel and marking as occupied if it is present in all silhouettes.
  - Stored the voxels in false colored pointcloud.
<p float="left">
<img src="https://github.com/ipsavani/3D-ComputerVision/blob/main/Voxel-Based%20Reconstruction/occupied_voxels1.png" width="300" height="300">
<img src="https://github.com/ipsavani/3D-ComputerVision/blob/main/Voxel-Based%20Reconstruction/occupied_voxels2.png" width="300" height="300">
<img src="https://github.com/ipsavani/3D-ComputerVision/blob/main/Voxel-Based%20Reconstruction/occupied_voxels3.png" width="300" height="300">
<p/>
<br/>

- part 2:
  - Extracted rgb values from each image for corresponding voxel
  - Calculated sum of squared differences wrt mean of intensity.
  - If voxel is photo consistent with at least 6 out of 8 images, voxel is marked as surface voxel and is assigned the rgb values of image with minimum SSD.
  - Surface voxels are stored in point cloud.
<p float="left">
<img src="https://github.com/ipsavani/3D-ComputerVision/blob/main/Voxel-Based%20Reconstruction/surface_voxels1.png" width="300" height="300">
<img src="https://github.com/ipsavani/3D-ComputerVision/blob/main/Voxel-Based%20Reconstruction/surface_voxels2.png" width="300" height="300">
<img src="https://github.com/ipsavani/3D-ComputerVision/blob/main/Voxel-Based%20Reconstruction/surface_voxels3.png" width="300" height="300">
<p/>
<br/>

 
 
