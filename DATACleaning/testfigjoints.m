% load('.\traintest\joints_train.mat');
% 
k = 12444;
% plot3(joints(1,:,k), joints(2,:,k), joints(3,:,k), '.');
% 
plot3(jointsALLSCALE(1,:,k), jointsALLSCALE(2,:, k), jointsALLSCALE(3,:, k), '.');

% plot3(joints_train(1,:,k), joints_train(2,:,k), joints_train(3,:,k), '.');

% k = 500;
% I = mat2gray(image_test(:,:,k), [0,1]);
% imshow(I);

% matfiles = dir('.\*.mat');
% 
% for i=1:length(matfiles)
%     load(matfiles(i).name);    
% end

