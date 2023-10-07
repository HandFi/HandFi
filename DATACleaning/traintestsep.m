csifile = ".\dataall\csi_allnorm.mat";
imagefile = ".\dataall\images.mat";
jointsfile = ".\dataall\joints.mat";

load(csifile);
% load(jointsfile);
% load(imagefile);
% image_train = imagef_all(:,:,1:4500);
% image_test = imagef_all(:,:,4501:5000);
% joints_train = jointsALLSCALE(:,:,1:4500);
% joints_test = jointsALLSCALE(:,:,4501:5000);
% joints_train_noscale = jointsALL(:,:,1:4500);
% joints_test_noscale = jointsALL(:,:,4501:5000);
csi_train = csi_all(:,:,:,1:4500);
csi_test = csi_all(:,:,:,4501:5000);

for i = [5001 10001 15001 20001 25001 30001 35001]
%     image_train = cat(3, image_train, imagef_all(:,:,i:i+4499));
%     image_test = cat(3, image_test, imagef_all(:,:,i+4500:i+4999));
%     joints_train = cat(3, joints_train, jointsALLSCALE(:,:,i:i+4499));
%     joints_test = cat(3, joints_test, jointsALLSCALE(:,:,i+4500:i+4999));
%     joints_train_noscale = cat(3, joints_train_noscale, jointsALL(:,:,i:i+4499));
%     joints_test_noscale = cat(3, joints_test_noscale, jointsALL(:,:,i+4500:i+4999));
    csi_train = cat(4, csi_train, csi_all(:,:,:,i:i+4499));
    csi_test = cat(4, csi_test, csi_all(:,:,:,i+4500:i+4999));
end


% save(".\dataall\images_train.mat", 'image_train');
% save(".\dataall\images_test.mat", 'image_test');
% save(".\dataall\joints_train.mat", 'joints_train', 'joints_train_noscale');
% save(".\dataall\joints_test.mat", 'joints_test', 'joints_test_noscale');
fprintf(mat2str(size(csi_train)) + "\n");
fprintf(mat2str(size(csi_test)) + "\n");
save(".\normcsiall\csi_train.mat", 'csi_train',  '-v7.3');
save(".\normcsiall\csi_test.mat", 'csi_test',  '-v7.3');
