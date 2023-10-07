matfiles = dir('.\L\*.jpg');

dates=[matfiles.datenum];
[~,order]=sort(dates);
matfiles=matfiles(order);

A = imread(strcat('.\L\', matfiles(1).name));
% fprintf("current image:" + int2str(1) + " \n");
A = imresize(A, [114,114]);
A_bw = A > 150;
image_all = A_bw;

for i=2:length(matfiles)
    A = imread(strcat('.\L\', matfiles(i).name));
    fprintf("current image:" + int2str(i) + " \n");
    A = imresize(A, [114,114]);
    A_bw = A > 150;
    image_all = cat(3, image_all, A_bw);
end

imageDataFile = ".\imageL.mat";
save(imageDataFile, 'image_all');