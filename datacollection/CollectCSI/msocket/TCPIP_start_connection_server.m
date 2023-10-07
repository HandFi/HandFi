% clear;
% addpath('./../msocket');
% srvsock 	= mslisten(6767);
% sock 		= msaccept(srvsock,10);
% ans         = msCSI_recv(sock,10);
% msclose(srvsock);
% clear srvsock;
% fprintf('Connection is built!\n');

% sock_vec = zeros(1,4,'int32');
% sock_vec(1,1) = 444;
% sock_vec(1,2) = 333;
% sock_vec(1,3) = 222;
% sock_vec(1,4) = 111;
% 
% sock_cnt    = length(sock_vec);
% sock_max    = max(sock_vec);
% sock_min    = min(sock_vec);
% sock_vec_out = msCSI_server( sock_cnt,sock_vec,sock_min,sock_max);
% cleanupObj = onCleanup(@cleanMeUp);

server_sock 	= mslisten(6767);

sock_vec_in     = zeros(1,1,'int32');

sock_vec_in(1,1)= server_sock;

sock_cnt_in    = length(sock_vec_in);
sock_max    = max(sock_vec_in);
sock_min    = min(sock_vec_in);
f1  = figure;
f2  = figure;
% f3  = figure;
fig_cell = {f1,f2};

tx_ant_num  = 1;
rx_ant_num  = 3;


csi_all_cell    = cell(500,2);
csi_cnt         = zeros(2,1);
while 1  
    
    [sock_vec_out,sock_cnt_out,CSI_struct] = msCSI_server_tmp(sock_cnt_in,sock_vec_in,sock_min,sock_max,1);          
    if (length(sock_vec_in) > 1)                        % we connect with at least 1 client
        if (length(CSI_struct) >= 1)                     % the output CSI structure must not be empty
            fprintf("timeStamp is:%f\n",CSI_struct(1,1).timestamp);
%             for csi_st_idx = 1:1:length(CSI_struct)     % all the structure
%                 for sock_idx = 2:1:length(sock_vec_in)
%                     
%                 end                
%             end
%             
%             for j = 2:1:length(sock_vec_in)     % we check every client socket
%                 for k = 1:1:length(CSI_struct)  % we check every CSI structure
%                     tic
%                     if isempty(CSI_struct(1,k)) % if the structure is empty leave it
%                         printf("empty\n");
%                         continue;
%                     end
%                     toc
%                     tic
%                     if CSI_struct(1,k).socket == sock_vec_in(j,1)
%                         csi_cnt(j-1,1) = csi_cnt(j-1,1) + 1;
%                         csi_all_cell{csi_cnt(j-1,1),j-1} = CSI_struct(1,k);
%                         fprintf("We plot\n");
%                         CSI  = CSI_struct(1,k).csi;
%                         CSI  = squeeze(CSI(:,1,:));
%     %                     figure(fig_cell{1,j-1})
%     %                     tic
%     %                     plotCSI(CSI); 
%     %                     toc
%                     end
%                     toc
%                 end
%             end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    sock_cnt_in     = sock_cnt_out;
    sock_vec_in     = sock_vec_out;
    sock_max        = max(sock_vec_in);
    sock_min        = min(sock_vec_in);    
end
for i = 1:1:length(sock_vec_out)
    fprintf("close all active socket and exit!!\n");
    msclose(sock_vec_out(i,1));
end
function plotCSI(CSI)
    amp_max     = 60;
    amp_min     = 0;
    ph_max      = 6;
    ph_min      = -6;
    CSI_a       = CSI(1,:);
    subplot(3,2,1); plot(db(abs(CSI_a))); 
    ylim([amp_min,amp_max]);
    xlabel('subcarriers');
    ylabel('Ampltude');drawnow    

    subplot(3,2,2); plot(unwrap(angle(CSI_a))); 
    ylim([ph_min,ph_max]);
    xlabel('subcarriers');
    ylabel('Phase');drawnow

    CSI_b       = CSI(2,:);
    subplot(3,2,3); plot(db(abs(CSI_b)));  
    ylim([amp_min,amp_max]);
    xlabel('subcarriers');
    ylabel('Ampltude');drawnow
    subplot(3,2,4); plot(unwrap(angle(CSI_b))); 
    ylim([ph_min,ph_max]);
    xlabel('subcarriers');
    ylabel('Phase');drawnow

    CSI_c       = CSI(3,:);
    subplot(3,2,5); plot(db(abs(CSI_c)));
    ylim([amp_min,amp_max]);
    xlabel('subcarriers');
    ylabel('Ampltude');drawnow
    subplot(3,2,6); plot(unwrap(angle(CSI_c)));
    ylim([ph_min,ph_max]);
    xlabel('subcarriers');
    ylabel('Phase');drawnow
end
function cleanMeUp()
    for i = 1:1:length(sock_vec_out)
        fprintf("close all active socket and exit!!\n");
        msclose(sock_vec_out(i,1));
    end
end