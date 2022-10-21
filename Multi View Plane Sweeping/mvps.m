% import images and camera data
img = imread('fountain_dense/urd/0005.png');
camdat = importdata('fountain_dense/kdata/0005.png.camera');
img1 = imread('fountain_dense/urd/0004.png');
camdat1 = importdata('fountain_dense/kdata/0004.png.camera');
img2 = imread('fountain_dense/urd/0006.png');
camdat2 = importdata('fountain_dense/kdata/0006.png.camera');

er_rep = true;
SAD = true;

K = camdat(1:3,:);
R = camdat(5:7,:);
t = camdat(8,:)';
R1 = camdat1(5:7,:);
t1 = camdat1(8,:)';
R2 = camdat2(5:7,:);
t2 = camdat2(8,:)';
dims = camdat(9,:);

% distance to sweep
z = 5;
Z = 10;
% number of planes
n_planes = 100;
% window size
win_size = 11;
% distance between each plane
plane_dist = (Z-z)/n_planes;

% resizing images to shorten computation time
img = im2double(imresize(img,0.2));
img1 = im2double(imresize(img1,0.2));
img2 = im2double(imresize(img2,0.2));

% readjusting camera intrinsic according to scale(0.2)
ax = 2*atan(K(1,3)/K(1,1));
ay = 2*atan(K(2,3)/K(2,2));
fx = ((dims(1)*0.2)/2)/(tan(ax/2));
fy = ((dims(2)*0.2)/2)/(tan(ay/2));
K = [((dims(1)*0.2)/2)/(tan(ax/2)) 0 ((dims(1)*0.2)/2);0 ((dims(2)*0.2)/2)/(tan(ay/2)) ((dims(2)*0.2)/2);0 0 1];

% extrinsic
R_1 = [R1 t1 ; [0 0 0 1]]\[R t; [0 0 0 1]];
R_2 = [R2 t2 ; [0 0 0 1]]\[R t; [0 0 0 1]];

R1 = R_1(1:3,1:3);
t1 = R_1(1:3,4);
R2 = R_2(1:3,1:3);
t2 = R_2(1:3,4);
n = R2*[0 0 -1]';
[r, c, ~] = size(img);

% store all plane sweep images in an array
trans_img1 = cell(1,n_planes+1);
trans_img2 = cell(1,n_planes+1);
for x = 1:n_planes
    trans_img1{x} = zeros(r,c);
    trans_img2{x} = zeros(r,c);
end

win_half = floor(win_size/2);
depth_map = zeros(r, c);



x = repmat(1:c,r,1);
y = repmat((1:r)',1,c);

plane_idx = 0;
for d = z:plane_dist:Z
    plane_idx = plane_idx + 1;
    % H matrix for left to reference image
    h_right = (K*(R1-((t1*n')/d)))/K;
    h_right = h_right/h_right(3,3);
    % H matrix for right to reference image
    h_left = (K*(R2-((t2*n')/d)))/K;
    h_left = h_left/h_left(3,3);
    xt = ((h_right(1,1).*x)+(h_right(1,2).*y))+h_right(1,3);
    yt = ((h_right(2,2).*y)+(h_right(2,1).*x))+h_right(2,3);
    s = ((h_right(3,1).*x)+(h_right(3,2).*y))+h_right(3,3);
    xt = xt./s;
    yt = yt./s;
    trans_img1{plane_idx} = interp2(x, y, 255*rgb2gray(img1), xt, yt, 'linear', 0);
    xt = ((h_left(1,1).*x)+(h_left(1,2).*y))+h_left(1,3);
    yt = ((h_left(2,2).*y)+(h_left(2,1).*x))+h_left(2,3);
    s = ((h_left(3,1).*x)+(h_left(3,2).*y))+h_left(3,3);
    xt = xt./s;
    yt = yt./s;
    trans_img2{plane_idx} = interp2(x, y, 255*rgb2gray(img2), xt, yt, 'linear', 0);      
end

% depth map using SAD and SSD
img = padarray(img,[win_half win_half]);
right = cell(1,length(trans_img1));
left = cell(1,length(trans_img1));

for d = 1:length(trans_img1)
    if SAD==true
        right{d} = abs(img-padarray(trans_img1{d},[win_half win_half]));
        left{d} = abs(img-padarray(trans_img2{d},[win_half win_half]));
    else
        right{d} = (img-padarray(trans_img1{d},[win_half win_half])).^2;
        left{d} = (img-padarray(trans_img2{d},[win_half win_half])).^2;
    end
end

for j = 1:c
    for i = 1:r
        dv = zeros(1, length(trans_img1));
        for d = 1:length(trans_img1)
            sad_r = sum(sum(right{d}(i:i+(win_size-1),j:j+(win_size-1))));
            sad_l = sum(sum(left{d}(i:i+(win_size-1),j:j+(win_size-1))));
            c = (sad_r+sad_l)/2;
            dv(d) = c;
        end
        [v, idx] = min(dv);
        depth_map(i,j) = -(((Z-z)/n_planes)*idx) / ([j i 1.0]*inv(K')*n);
    end
end


figure;
title('Depth Map');
imshow(uint8(depth_map*16));
imwrite(uint8((depth_map*16)),'fountain_dense_depthmap.jpg');

if er_rep == true
    % Error Map
    load data.mat
    bgpoints = BackgroundPointCloudRGB(1:3,:);
    fgpoints = ForegroundPointCloudRGB(1:3,:);
    points = [bgpoints fgpoints];
    points(4,:) = 1;
    
    K = [K [0 0 0]'];
    ones = [1 1 1 1];
    P = K*diag(ones);
    
    pts_2d = P*points;
    pts_2d(1,:) = pts_2d(1,:)./pts_2d(3,:); 
    pts_2d(2,:) = pts_2d(2,:)./pts_2d(3,:);
    
    temp = zeros(pts_2d(1,end),pts_2d(2,end));
    
    for i = 1:length(pts_2d)
        temp(round(pts_2d(2,i)),round(pts_2d(1,i))) = pts_2d(3,i);
    end
    
    temp = im2double(imresize(temp,0.2));
    [r, c, ~] = size(temp);
    
    error_map = abs(temp - depth_map);
    avg_px_error = (sum(sum(error_map)))/(r*c);
    disp(avg_px_error);

    imwrite( ind2rgb(im2uint8(mat2gray(error_map)), parula(256)), 'error_map.jpg')
end