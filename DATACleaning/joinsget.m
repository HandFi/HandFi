% try to combine all the joints csv to one mat

matfiles = dir('.\L\*.csv');

dates=[matfiles.datenum];
[~,order]=sort(dates);
matfiles=matfiles(order);

joints = [];
for i=1:length(matfiles)
    csv = csvread(strcat('.\L\', matfiles(i).name),1);
    for j=1:size(csv,1)
        row = csv(j, :);
        joint = reshape(row,22,3);
        joint = joint.';
        joints = cat(3, joints, joint);
    end
end

jointsDataFile = ".\jointsL.mat";
save(jointsDataFile, 'joints');


