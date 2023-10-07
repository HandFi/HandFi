csifile = ".\dataall\csi_all.mat";
imagefile = ".\dataall\images.mat";
jointsfile = ".\dataall\joints.mat";

% load(csifile);
load(jointsfile);
load(imagefile);
image_data = imagef_all(:,:,1:1000);
joints_data = jointsALLSCALE(:,:,1:1000);
% csi_data = csi_all(:,:,:,1:1000);

for i = [5001 10001 15001 20001 25001 30001 35001]
    image_data = cat(3, image_data, imagef_all(:,:,i:i+999));
    joints_data = cat(3, joints_data, jointsALLSCALE(:,:,i:i+999));
%     csi_data = cat(4, csi_data, csi_all(:,:,:,i:i+999));
end


save(".\image_data.mat", 'image_data');
save(".\joints_data.mat", 'joints_data');
% fprintf(mat2str(size(csi_data)) + "\n");
% save(".\csi_data.mat", 'csi_data',  '-v7.3');

