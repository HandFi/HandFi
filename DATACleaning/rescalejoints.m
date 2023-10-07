matfiles = dir('.\*.mat');
jointsALL = [];
for i=1:length(matfiles)
    load(matfiles(i).name, 'joints');
    jointsALL = cat(3, jointsALL, joints);
end

jointsALLSCALE = rescale(jointsALL, 0, 1);
save(".\joints.mat", 'jointsALL' , 'jointsALLSCALE');

% matfiles = dir('.\B\*.csv');
% csv = csvread(strcat('.\B\', matfiles(1).name),1);
% rowB = csv(2, :);
% rowB = reshape(rowB,3,21);
% 
% matfiles = dir('.\C\*.csv');
% csv = csvread(strcat('.\C\', matfiles(1).name),1);
% rowC = csv(2, :);
% rowC = reshape(rowC,3,21);
% 
% matfiles = dir('.\D\*.csv');
% csv = csvread(strcat('.\D\', matfiles(1).name),1);
% rowD = csv(2, :);
% rowD = reshape(rowD,3,21);
% 
% matfiles = dir('.\G\*.csv');
% csv = csvread(strcat('.\G\', matfiles(1).name),1);
% rowG = csv(2, :);
% rowG = reshape(rowG,3,21);
% 
% matfiles = dir('.\Y\*.csv');
% csv = csvread(strcat('.\Y\', matfiles(1).name),1);
% rowY = csv(2, :);
% rowY = reshape(rowY,3,21);

% rows = cat(3,rowB, rowC, rowD, rowG, rowY);