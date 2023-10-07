% video input 1 is the built in webcam
% video input 2 is the usb camera
clear cam
cam = videoinput('winvideo',2);
cam
savepath='';
nametemplate='image';
format = '.jpg';
imnum = 0;

for K = 1:10
    if(mod(K,2)==0)

        filename = [nametemplate int2str(imnum) format];
        fullpath = fullfile(savepath, filename);
        imwrite(img, fullpath);
        imnum = imnum + 1;        
    end
    pause(1);
end

clear cam
% preview(cam) to see
% snapshot(cam) to take a snapshot
