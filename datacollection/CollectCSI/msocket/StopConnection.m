for i = 1:1:length(sock_vec_out)
    fprintf("close all active socket and exit!!\n");
    msclose(sock_vec_out(i,1));
end