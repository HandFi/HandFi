matfiles = dir('.\L\*.mat');

dates=[matfiles.datenum];
[~,order]=sort(dates);
matfiles=matfiles(order);

csidata_all = [];

for i=1:length(matfiles)
    load(strcat('.\L\', matfiles(i).name), 'csi_data');
    csidata_all = cat(3, csidata_all, csi_data);
end

csia_1 = csidata_all(1,:,:);
csia_2 = csidata_all(2,:,:);
csia_3 = csidata_all(3,:,:);
csia_1 = squeeze(csia_1(1,:,:));
csia_2 = squeeze(csia_2(1,:,:));
csia_3 = squeeze(csia_3(1,:,:));

R1 = real(csia_1); 
I1 = imag(csia_1);
R2 = real(csia_2); 
I2 = imag(csia_2);
R3 = real(csia_3); 
I3 = imag(csia_3);

csi_datari = cat(3, R1, I1, R2, I2, R3, I3);

csi_datari = permute(csi_datari,[3 2 1]);

l = length(csi_datari(1,:,1))/20;

csi_tall = [];
for i=1:length(csi_datari(1,:,1))/20
    start = ((i-1)*20+1);
    fprintf(int2str(start)+ " \n");
    e = i*20;
    csi_t = csi_datari(:,start:e,:);
    csi_tall = cat(4, csi_tall, csi_t);
end

fprintf(mat2str(size(csi_tall)) + "\n");

csiDataFile = ".\csiL.mat";
save(csiDataFile, 'csi_tall');


% A = [1,2,13;3,4,14];
% B = [5,6,15;7,8,16];
% C = [9,10,17;11,12,18];
% D = [19,20,21;22,23,24];
% Z = cat(3,A,B,C,D);
% Y = permute(Z,[3 2 1]);