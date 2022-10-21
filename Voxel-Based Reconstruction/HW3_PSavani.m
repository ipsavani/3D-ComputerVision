rawP = [ 776.649963  -298.408539 -32.048386  993.1581875 132.852554  120.885834  -759.210876 1982.174000 0.744869  0.662592  -0.078377 4.629312012;
    431.503540  586.251892  -137.094040 1982.053375 23.799522   1.964373    -657.832764 1725.253500 -0.321776 0.869462  -0.374826 5.538025391;
    -153.607925 722.067139  -127.204468 2182.4950   141.564346  74.195686   -637.070984 1551.185125 -0.769772 0.354474  -0.530847 4.737782227;
    -823.909119 55.557896   -82.577644  2498.20825  -31.429972  42.725830   -777.534546 2083.363250 -0.484634 -0.807611 -0.335998 4.934550781;
    -715.434998 -351.073730 -147.460815 1978.534875 29.429260   -2.156084   -779.121704 2028.892750 0.030776  -0.941587 -0.335361 4.141203125;
    -417.221649 -700.318726 -27.361042  1599.565000 111.925537  -169.101776 -752.020142 1982.983750 0.542421  -0.837170 -0.070180 3.929336426;
    94.934860   -668.213623 -331.895508 769.8633125 -549.403137 -58.174614  -342.555359 1286.971000 0.196630  -0.136065 -0.970991 3.574729736;
    452.159027  -658.943909 -279.703522 883.495000  -262.442566 1.231108    -751.532349 1884.149625 0.776201  0.215114  -0.592653 4.235517090];
% Read the projection matrices
P = zeros(3,4,8);
for i=1:8
    for j=1:3
        P(j,1:4,i) = rawP(i,4*(j-1)+1:4*(j-1)+4);
    end
end
% read the RGB images
imgs = zeros(582, 780, 3, 8);
for i=1:8
    img = imread("cam0"+int2str(i-1)+"_00023_0000008550.png");
    imgs(:,:,:,i) = img;
end
% Read the silhouette images
sils = zeros(582, 780, 8);
for i=1:8
    sil = imread("silh_cam0"+int2str(i-1)+"_00023_0000008550.pbm");
    sils(:,:,i) = sil;
end

% part 1
% mat to store selected voxels
occupied_voxs = zeros(2000000,3);
% set resolution (number of voxels)
res = 100000000;
% calculate single voxel length(volume of grid/resolution)
% used a tighter initial grid of x=4, y=5 and z=2.5
step = nthroot((4*5*2.5)/res,3);
no_of_occ_voxs = 0;
% iterate through each voxel
for x = -2:step:2
    for y = -2.5:step:2.5
        for z = 0:step:2.5
            % initialize voxel position in world coordinates
            X_world = [x ;y ;z ;1.0];
            % variable to check if voxel is a silhouette
            is_silhouette = 0;
            for i = 1:8
                % calculate projected coordinates for voxel
                X_img = P(:,:,i)*X_world;
                % scale z to 1, and convert coordinates to image indexes by
                % adding 1
                X_img = round(X_img/X_img(3))+1;
                % check if projected voxel is inside image
                if (0<X_img(1)/780) && (X_img(1)/780<=1) && (0<X_img(2)/582) && (X_img(2)/582<=1)
                    % add silhouette value
                    is_silhouette = is_silhouette + sils(X_img(2),X_img(1),i);
                end
            end
            % add voxel to occupied list if it projects in all silhouette
            % images
            if is_silhouette == 8
                no_of_occ_voxs=no_of_occ_voxs+1;
                occupied_voxs(no_of_occ_voxs,:) = [x y z];
            end
        end
    end
end

% write occupied voxels to pointcloud
ptc = pointCloud(occupied_voxs(1:no_of_occ_voxs,:));
pcwrite(ptc,'occupied_voxels','PLYFormat','ascii');
occ_pc = pcread('occupied_voxels.ply');
% occ_pc.Count
pcshow(occ_pc);
%% 
% part 2 and 3
% mat to store selected surface voxels
surface_voxs = zeros(size(occupied_voxs(1:no_of_occ_voxs,:)));
% mat to store colors
sur_colors = zeros(size(occupied_voxs(1:no_of_occ_voxs,:)));
threshold=500;
no_sur_voxs=0;
% iterate through all occupied voxels
for v=1:size(occupied_voxs(1:no_of_occ_voxs,:),1)
    % get world cordinates of the voxel
    x = occupied_voxs(v,1);y=occupied_voxs(v,2);z = occupied_voxs(v,3) ;
    X_world = [x ;y ;z ;1.0];
    rgb = zeros(8,3);
    for i=1:8
        % calculate image coordinates of the voxel
        X_img = P(:,:,i)*X_world;
        X_img = round(X_img/X_img(3))+1;
        % get rgb values from projected image points
        r = imgs(X_img(2), X_img(1), 1, i);
        g = imgs(X_img(2), X_img(1), 2, i);
        b = imgs(X_img(2), X_img(1), 3, i);
        rgb(i,:) = [r g b];
    end
    ssd = zeros(1,8);
    % calculate mean intensity of same pixel in all 8 images(rgb/3*8)
    mean_intensity = sum(sum(rgb))/24;
    % calculate sum of squared differences for the intensities with the
    % mean
    for i = 1:8
        ssd(i)=((sum(rgb(i,:))/3)-mean_intensity)^2;
    end
    %get minimum SSD
    [m,idx] = min(ssd);
%     cost=sum(ssd)/8;
    count = sum(ssd>threshold);
    
    if count>=6
        no_sur_voxs=no_sur_voxs+1;
        % if atleast count number of images are photoconsistent, add voxel as
        % surface voxel
        surface_voxs(no_sur_voxs,:)=[x y z];
        % select the colors from image with minimum SSD as voxel color
        sur_colors(no_sur_voxs,:)=rgb(idx,:);
    end
end

% write surface voxels to pointcloud
ptc = pointCloud(surface_voxs(1:no_sur_voxs,:));
ptc.Color = uint8(sur_colors(1:no_sur_voxs,:));
pcwrite(sur_vox_mat_ptc,'surface_voxels','PLYFormat','ascii');
surf_pc = pcread('surface_voxels.ply');
pcshow(surf_pc);
