matfiles = dir('.\*.mat');

load(matfiles(1).name, 'image_all');
imagef_all = image_all;

for i=2:length(matfiles)
    load(matfiles(i).name, 'image_all');
    imagef_all = cat(3, imagef_all, image_all);
end

save(".\images.mat", 'imagef_all');