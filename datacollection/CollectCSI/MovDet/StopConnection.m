 
for i = 1:1:length(sock_vec_out)
    fprintf('close all active socket and exit!!\n');
    msclose(sock_vec_out(i,1));
end
% 
% csiDataFile = "/home/ji/Desktop/CollectJoints/dataset/B/";
% F = sprintf('csidata_%s.mat',datetime('now','Format',"MM-dd-HH-mm-ss"));
% csi_data = cat(3, csi_idx1_all, csi_idx1_all(:,:,(2*end-4559):end));
% if size(csi_data,3) > 4560 
%     csi_data(:,:, 4561:end) = [];
% end
% save(fullfile(csiDataFile,F),'csi_data');

csiDataFile = "/home/ji/Desktop/CollectJoints/datasetRight/H/";
F = sprintf('csidata_%s.mat',datetime('now','Format',"MM-dd-HH-mm-ss"));
csi_data = cat(3, csi_idx1_all, csi_idx1_all(:,:,(2*end-3999):end));
if size(csi_data,3) > 4000 
    csi_data(:,:, 4001:end) = [];
end

% csiDataFile = "/home/ji/Desktop/CollectJoints/datasetTRACK/P/";
% F = sprintf('csidata_%s.mat',datetime('now','Format',"MM-dd-HH-mm-ss"));
% csi_data = cat(3, csi_idx1_all, csi_idx1_all(:,:,(2*end-299):end));
% if size(csi_data,3) > 300 
%     csi_data(:,:, 301:end) = [];
% end


save(fullfile(csiDataFile,F),'csi_data');

% save(fullfile(csiDataFile,F),'timeall','csi_idx1_all','rssi0','rssi1','rssi2');

msclose(sockWserver);

rmpath('./../../MatlabCode/xDTrack-2D-Time');
rmpath('./../../MatlabCode/xDTrack-2D-Time_sm');

