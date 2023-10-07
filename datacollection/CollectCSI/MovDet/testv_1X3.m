clear;
addpath('/home/ji/Desktop/CollectCSI/msocket/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Estimation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

car_num     = 114;
MAC_idx_init = 24525;
mm = ones(1,114);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   SOCKET  Processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% serverIP    = '155.69.142.136';
% sockWserver = msconnect(serverIP,6767);
% fprintf('Connection with server is built!\n');

server_sock 	= mslisten(6767);
sock_vec_in     = zeros(1,1,'int32');
sock_vec_in(1,1)= server_sock;

sock_cnt_in = length(sock_vec_in);
sock_max    = max(sock_vec_in);
sock_min    = min(sock_vec_in);


CSI_NUM_V   = 100;  % the number of CSI we will record
csi_num     = 0;    % count the number of csi received
idx1_num    = 0;
idx2_num    = 0;
idx3_num    = 0;

CSI_st_all  = cell(1,CSI_NUM_V);
CSI_M       = zeros(9,car_num);
CSI_M_all   = zeros(9,car_num,CSI_NUM_V);

csi_idx_v   = 1:1:CSI_NUM_V;
% csi_idx_v   = repmat(csi_idx_v,path,1);

csia_1 = zeros(CSI_NUM_V,car_num);
csia_2 = zeros(CSI_NUM_V,car_num);
csia_3 = zeros(CSI_NUM_V,car_num);

% en1 = zeros(300,1);
% en2 = zeros(300,1);
% en3=  zeros(300,1);
% ans1 = zeros(300,1);
% ans2 = zeros(300,1);
% ans3 = zeros(300,1);
% ans4 = zeros(300,1);
% sttttd = zeros(300,3);

% p1 = zeros(300,1);
% p2 = zeros(300,1);
% p3 = zeros(300,1);
%  ans1(1,1) = 0;
%  ans2(1,1) = 0;
%  ans3(1,1) = 0;
% sttttd =[];
index = 0;
jitter = 0;
CSI_M_calib_all   = zeros(9,car_num,CSI_NUM_V);
csi_idx1_all   = zeros(3,car_num,CSI_NUM_V);
csi_idx2_all   = zeros(3,car_num,CSI_NUM_V);
csi_idx3_all   = zeros(3,car_num,CSI_NUM_V);
rssi0 = zeros(300,1);
rssi1 = zeros(300,1);
rssi2 = zeros(300,1);
timeall =  zeros(300,1);
pkt = zeros(300,1);
LEN = zeros(300,1);
aband = 0;
rateall = zeros(300,1);
%%%%% initial GUI %%%%%%%%%%%


while 1     
    tic
    [sock_vec_out,sock_cnt_out,CSI_struct] = msCSI_server_tmp(sock_cnt_in,sock_vec_in,sock_min,sock_max,2); 
    if (length(sock_vec_in) > 1)                        % we connect with at least 1 client
       
        if (length(CSI_struct) >= 1)                     % the output CSI structure must not be empty
        %    for csi_st_idx = 1:1:length(CSI_struct)     % all the structure

        
                CSI_entry   = CSI_struct(1,1);
                N_tx        = CSI_entry.nc;
                N_rx        = CSI_entry.nr;    
                num_tones   = CSI_entry.num_tones;
                pkt_idx     = CSI_entry.pkt_idx;
                MAC_idx     = CSI_entry.MAC_idx;
                payload     = CSI_entry.payload_len;
                rss0        = CSI_entry.rssi_0;
                rss1        = CSI_entry.rssi_1;
                rss2        = CSI_entry.rssi_2;
                time        = CSI_entry.timestamp;
                len         = CSI_entry.csi_len;
                rate        = CSI_entry.rate;
                PHY         = CSI_entry.phyerr;
                
                if MAC_idx ~= MAC_idx_init           % filter
                    continue;
                end
                
                if PHY ~= 0           % filter
                    continue;
                end
                
                if N_rx < 3  || num_tones~= car_num
                    continue;
                end
                
                CSI_ori     = CSI_entry.csi;
                
                if isempty(CSI_ori)
                    continue;
                end
                
                if payload ~= 120          % filter
                    continue;
                end
                
                if  csi_num > 5 && rss0 < rssi0(csi_num-1,1) - 8
                    jitter = jitter + 1;
                    continue;
                end
                %%%%%%%%%%%%%%%%%%%%%%%%
                
                CSIRX1     = squeeze(CSI_ori(:,1,:));
%                 CSIRX2     = squeeze(CSI_ori(:,2,:));
%                 CSIRX3     = squeeze(CSI_ori(:,3,:));
                
                if  find (CSIRX1 == 0) 
                    aband = aband + 1 ;
                    continue;
                end
                
                csi_num = csi_num + 1;
                rssi0(csi_num,1) = rss0;
                rssi1(csi_num,1) = rss1;
                rssi2(csi_num,1) = rss2;
                timeall(csi_num,1) = time;
                pkt(csi_num,1) = pkt_idx;
                LEN(csi_num,1) = len;
                rateall(csi_num,1) = rate;
                
                csi_idx1 = CSIRX1;
                csi_idx1_all(:,:,csi_num)   = csi_idx1;  %reserved all csi
%                 csi_idx2 = CSIRX2;
%                 csi_idx2_all(:,:,csi_num)   = csi_idx2;  %reserved all csi
%                 
%                 csi_idx3 = CSIRX3;
%                 csi_idx3_all(:,:,csi_num)   = csi_idx3;  %reserved all csi
                
                %%%%%%% obtain each antenna csi%%%%%%%%%%
                csia_1(csi_num,:) = CSIRX1(1,:);
                csia_2(csi_num,:) = CSIRX1(2,:);
                csia_3(csi_num,:) = CSIRX1(3,:);
%                fprintf("Current packet " + int2str(packetCount) + " \n");

                
                [x1, x2, x3] = dy([csia_1(csi_num,:);csia_2(csi_num,:);csia_3(csi_num,:)]);
                nCSIA1(csi_num, 1) = x1;
                nCSIA2(csi_num, 1) = x2;
                nCSIA3(csi_num, 1) = x3;
                
%                plot(nCSIA1,'-r');hold on; hold off;
%                xlabel('time');ylabel('amp(db)');drawnow;
                        
                
%                 tic
%                 if(mod(packetCount,10)==0)
%                     filename = [nametemplate int2str(imnum) format];
%                     fullpath = fullfile(imgsavepath, filename);
%                     if(packetCount==0)
%                         img = snapshot(cam);
%                     end
%                     img = snapshot(cam);
%                     imwrite(img, fullpath);
%                     imnum = imnum + 1;
%                 end
%                 packetCount = packetCount+1;
%                 toc
              %fprintf('CSI 1,1 --> Amplitude: %0.2f\tPhase Angle: %0.2f\t', round(abs(csia_1(packetCount,1)),2), round(angle(csia_1(packetCount,1)),2));
%               fprintf('CSI 1,1 --> Amplitude: %0.2f\tPhase Angle: %0.2f\t', round(real(csia_1(packetCount,1)),2), round(imag(csia_1(packetCount,1)),2));
%               packetCount = packetCount+1;
                
%                    plot(abs(csia_1(csi_num,:)),'-r','LineWidth',1.5);xlim([0,car_num]);hold on;       
%                 plot(p2,'-b');hold on;
%                 plot(p3,'-g');hold on;
             %   xlabel('time index');ylabel('angle(�)');  
             %   hold off;
              %  drawnow;  
           % end
%                    
% %         
                    
        end
    end
     %    save(saveFile,'csia_1','csia_2','csia_3');
      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%            adjust the sockets 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sock_cnt_in     = sock_cnt_out;
    sock_vec_in     = sock_vec_out;
    sock_max        = max(sock_vec_in);
    sock_min        = min(sock_vec_in);    
   
    
    toc     
 end

%save(saveFile,'csia_1','csia_2','csia_3');

