k = 400;

% load('test.mat');
% A = imresize(img_list(:,:,k), [224,224]);
% A_bw = A > 150;
I = mat2gray(image_all(:,:,k), [0,1]);
% I = mat2gray(image_test(:,:,k), [0,1]);
imshow(I);
