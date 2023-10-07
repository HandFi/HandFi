% Domain generalization
%iou
%cdata = [0.98 0.44 0.47 0.62 0.68; 0.43 0.96 0.61 0.56 0.50; 0.46 0.54 0.96 0.44 0.43; 0.78 0.47 0.50 0.97 0.81; 0.81 0.46 0.52 0.80 0.97;];
%mpjpe
cdata = [1.87 3.78 4.10 2.87 2.99; 3.69 2.04 3.03 3.56 3.48; 3.77 2.78 2.02 3.24 3.33; 2.88 3.04 3.12 2.02 2.77; 2.80 2.98 3.04 2.43 2.03;];
%xvalues = {'Small','Medium','Large'};
%yvalues = {'Green','Red','Blue','Gray'};
h = heatmap(cdata);

color1 = [88 83 159]/255; 
color2 = [137 135 186]/255;
color3 = [187 187 214]/255; % purple

color6 = [238 186 187]/255; 
color5 = [227 145 145]/255; 
color4 = [216 105 103]/255; 

position = [0 1/2 1];

resolution = 512;

% purple
my_colormap1 = interp1(position, [color1; color2; color3], linspace(0, 1, resolution));
% pink
my_colormap2 = interp1(position, [color4; color5; color6], linspace(0, 1, resolution));

% color1 = [88 83 159]/255; 
% color2 = [137 135 186]/255;
% color3 = [187 187 214]/255;  
% color4 = [238 186 187]/255; 
% % color4 = [227 145 145]/255; 
% color5 = [216 105 103]/255; 
% 
% position = [0 1/4 1/2 3/4 1];
% 
% resolution = 256;
% 
% my_colormap = interp1(position, [color1; color2; color3; color4; color5], linspace(0, 1, resolution));


h.Colormap = my_colormap2;

%h.Title = 'T-Shirt Orders';
h.XLabel = 'Test';
h.YLabel = 'Train';
%mycolor=[88,83,159;187,187,214;229,224,234;216,105,103;238,186,187];
%mycolor = mycolor./255;
%colormap(gca,mycolor);
set(gcf,'color','w');
%set(l, 'horizontalAlignment', 'right')
title('')
%set(gca, 'LooseInset', [0.07,0,0.03,0]);
%set(gca, 'linewidth', 1.5);

