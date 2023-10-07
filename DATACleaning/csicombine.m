matfiles = dir('.\*.mat');

csi_all = [];

for i=1:length(matfiles)
    load(matfiles(i).name, 'csi_tall');
    csi_all = cat(4, csi_all, csi_tall);
end

fprintf(mat2str(size(csi_all)) + "\n");

save(".\csi_allnorm.mat", 'csi_all',  '-v7.3');
