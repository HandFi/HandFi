fileloc = 'C:\Users\Timothy-pc\Downloads\Powerpoint Samples\';
action = 'open_hand_distance\';
load([fileloc, action, 'csi_data.mat']);
writematrix(csia_1, [fileloc,action,'csi1.csv']);
writematrix(csia_2, [fileloc,action,'csi2.csv']);
writematrix(csia_3, [fileloc,action,'csi3.csv']);
