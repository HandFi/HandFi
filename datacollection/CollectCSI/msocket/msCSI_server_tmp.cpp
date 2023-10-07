#include <mex.h>
#include <math.h>

#include <matvar.h>

#if !defined(WIN32)
#include <sys/socket.h>
#include <unistd.h>
#include <sys/select.h>
#include <netdb.h>
#include <arpa/inet.h>
#else
#include <winsock2.h>
#endif

// csi status length
#define csi_st_len 23
#define TONE_40M 114 
#define BITS_PER_BYTE 8 
#define BITS_PER_COMPLEX_SYMBOL (2 * BITS_PER_SYMBOL) 
#define BITS_PER_SYMBOL      10
int checkCPUendian(){
	int num = 1;
	if(*((char*)&num) == 1){
	//	printf("little-endian\n");
		return 0;		
	}else{
	//	printf("big-endian\n");
		return 1;
	}
}
typedef struct {     
	int real;     
	int imag; 
}COMPLEX;

int signbit_convert(int data, int maxbit) {     
	if (data & (1 << (maxbit - 1)))    
	{ /*  negative */         
		data -= (1 << maxbit);     
	}     
	return data; 
}
void fill_csi_struct(unsigned char *buf_addr, mxArray *outCell,int cnt,int activeSockNum,int fd){
    int real, imag;
    int h_data, h_idx;
	int bitmask, current_data;
    int bits_left, nc_idx, nr_idx;

	double AntEng[9] = {0,0,0,0,0,0,0,0,0}; // maximum 3x3 antenna
	double scale_factor[9] = {0,0,0,0,0,0,0,0,0}; // maximum 3x3 antenna
	int    rssi_vec[3] = {0,0,0};    
	int k, idx;

	uint64_T	timestamp_int;

	const mwSize ArraySize[]  	= {1};

	mxArray 	*timestamp = mxCreateNumericArray(1,ArraySize,mxUINT64_CLASS,mxREAL);

	timestamp_int 	= ((buf_addr[0] & 0x00000000000000ff) << 56) | ((buf_addr[1] & 0x00000000000000ff) << 48) | 
			  	((buf_addr[2] & 0x00000000000000ff) << 40) | ((buf_addr[3] & 0x00000000000000ff) << 32)  |
			  	((buf_addr[4] & 0x00000000000000ff) << 24) | ((buf_addr[5] & 0x00000000000000ff) << 16)  |
			  	((buf_addr[6] & 0x00000000000000ff) << 8)  | ((buf_addr[7] & 0x00000000000000ff) << 0);

	uint64_T * ptrTime 	= (uint64_T *)mxGetPr(timestamp);
	*ptrTime 	= timestamp_int;

	int csi_len     = ((buf_addr[8] << 8) & 0xff00) | (buf_addr[9] & 0x00ff);
	int channel 	= ((buf_addr[10] << 8) & 0xff00) | (buf_addr[11] & 0x00ff);
	int buf_len 	= ((buf_addr[cnt-2] << 8) & 0xff00) | (buf_addr[cnt-1] & 0x00ff);
	
	int payload_len = ((buf_addr[csi_st_len] << 8) & 0xff00) | ((buf_addr[csi_st_len + 1]) & 0x00ff);
	if (buf_len == 641){
        payload_len = 0;
    }
	int phyerr  	= buf_addr[12];
	int noise 	= buf_addr[13];
	int rate    	= buf_addr[14];
	int chanBW 	= buf_addr[15];
	int num_tones 	= buf_addr[16];
	int nr 	= buf_addr[17];
	int nc 	= buf_addr[18];
   
    
	int rssi 	= buf_addr[19];
	int rssi_0 	= buf_addr[20];
	int rssi_1 	= buf_addr[21];
	int rssi_2 	= buf_addr[22];
    
	rssi_vec[0] 	= rssi_0;
	rssi_vec[1] 	= rssi_1;
	rssi_vec[2] 	= rssi_2;
		
	//unsigned char *local_h  = buf_addr + csi_st_len;
	unsigned char *local_h  = &buf_addr[25];

/*   	for(i=0;i<(int)csi_len;i++){
		print_tmp = local_h[i] & 0xFF;
		printf("0x%02x ",print_tmp);
	}
	printf("\n");
*/	

	const mwSize size[]  	= {nr, nc, num_tones};
	mxArray *csi  	= mxCreateNumericArray(3, size, mxDOUBLE_CLASS, mxCOMPLEX);
	double * ptrR 	=(double *)mxGetPr(csi);     
	double * ptrI 	=(double *)mxGetPi(csi);
	
	bits_left = 16; /* process 16 bits at a time */

    	/* 10 bit resoluation for H real and imag */
    	bitmask = (1 << BITS_PER_SYMBOL) - 1;
    	idx = h_idx = 0;
    	h_data = local_h[idx++];
    	h_data += (local_h[idx++] << BITS_PER_BYTE);
    	current_data = h_data & ((1 << 16) - 1); /* get 16 LSBs first */
	imag = current_data & bitmask;
   
	for (k = 0; k < num_tones; k++){
		for (nc_idx = 0; nc_idx < nc; nc_idx++) {
			for (nr_idx = 0; nr_idx < nr; nr_idx++) {
				if ((bits_left - BITS_PER_SYMBOL) < 0) {
					/* get the next 16 bits */
					h_data = local_h[idx++];
					h_data += (local_h[idx++] << BITS_PER_BYTE);
					current_data += h_data << bits_left;
					bits_left += 16;
				}
				imag = current_data & bitmask;
				
				imag = signbit_convert(imag, BITS_PER_SYMBOL);
				*ptrI = (double) imag;
				++ptrI;
				bits_left -= BITS_PER_SYMBOL;
				/* shift out used bits */
				current_data = current_data >> BITS_PER_SYMBOL; 
 				
				if ((bits_left - BITS_PER_SYMBOL) < 0) {
					/* get the next 16 bits */
					h_data = local_h[idx++];
					h_data += (local_h[idx++] << BITS_PER_BYTE);
					current_data += h_data << bits_left;
					bits_left += 16;
				}
				real = current_data & bitmask;
				
				real = signbit_convert(real, BITS_PER_SYMBOL);
				*ptrR = (double) real;
				++ptrR;
				bits_left -= BITS_PER_SYMBOL;
				
				/* shift out used bits */
				current_data = current_data >> BITS_PER_SYMBOL;
				AntEng[nc_idx*nr + nr_idx] += sqrt(pow(double(imag),2) + pow(double(real),2));
                //mexPrintf("nc_idx*nc+nr_idx: %d | ",nc_idx*nr + nr_idx);
			}
		}
        //mexPrintf("\n");
	}
	ptrR 	=(double *)mxGetPr(csi);     
	ptrI 	=(double *)mxGetPi(csi);
    
	for (nr_idx = 0; nr_idx < nr; nr_idx++){				
        for (nc_idx = 0; nc_idx < nc; nc_idx++) {
            AntEng[nc_idx*nr+nr_idx] = AntEng[nc_idx*nr+nr_idx] / num_tones;
            scale_factor[nc_idx*nr+nr_idx] = pow(10,double(rssi)/20)/AntEng[nc_idx*nr+nr_idx];
            for (k = 0; k < num_tones; k++){			
				ptrR[k*nr*nc+nc_idx*nr+nr_idx] = ptrR[k*nr*nc+nc_idx*nr+nr_idx]	* scale_factor[nc_idx*nr+nr_idx];
				ptrI[k*nr*nc+nc_idx*nr+nr_idx] = ptrI[k*nr*nc+nc_idx*nr+nr_idx]	* scale_factor[nc_idx*nr+nr_idx];
			}
		}
	}
    /*for (nr_idx = 0; nr_idx < nr; nr_idx++){
		AntEng[nr_idx] = AntEng[nr_idx] / (nc*num_tones);
		scale_factor[nr_idx] = pow(10,double(rssi_vec[nr_idx])/20)/AntEng[nr_idx];
		for (k = 0; k < num_tones; k++){
			for (nc_idx = 0; nc_idx < nc; nc_idx++) {
				ptrR[k*nr*nc+nc_idx*nr+nr_idx] = ptrR[k*nr*nc+nc_idx*nr+nr_idx]	* scale_factor[nr_idx];
				ptrI[k*nr*nc+nc_idx*nr+nr_idx] = ptrI[k*nr*nc+nc_idx*nr+nr_idx]	* scale_factor[nr_idx];
			}
		}
	}*/
    int pkt_idx     = ((buf_addr[csi_st_len+csi_len+71] << 8) & 0xff00) | (buf_addr[csi_st_len+csi_len+70] & 0x00ff);
    int MAC_idx     = ((buf_addr[csi_st_len+csi_len+17] << 8) & 0xff00) | (buf_addr[csi_st_len+csi_len+16] & 0x00ff);
	//mexPrintf("RSSI_0: %d   | RSSI_1: %d 	| RSSI_2: %d\n",rssi_0,rssi_1,rssi_2);
	//mexPrintf("RSSI_l: %f   | RSSI_l: %f 	| RSSI_l: %f\n",pow(10,double(rssi_vec[0])/20),pow(10,double(rssi_vec[1])/20),pow(10,double(rssi_vec[2])/20));
	
    //mexPrintf("AntEng_0: %f | AntEng_1: %f  | AntEng_2: %f\n",AntEng[0],AntEng[1],AntEng[2]);
    //mexPrintf("AntEng_3: %f | AntEng_4: %f  | AntEng_5: %f\n",AntEng[3],AntEng[4],AntEng[5]);
    //mexPrintf("AntEng_6: %f | AntEng_7: %f  | AntEng_8: %f\n",AntEng[6],AntEng[7],AntEng[8]);
	//mexPrintf("AntEng_0: %f | AntEng_1: %f  | AntEng_2: %f\n",20*log10(AntEng[0]),20*log10(AntEng[1]),20*log10(AntEng[2]));
	//mexPrintf("scaleFactor_0: %f | scaleFactor_1: %f  | scaleFactor_2: %f\n",scale_factor[0],scale_factor[1],scale_factor[2]);
    //mexPrintf("scaleFactor_3: %f | scaleFactor_4: %f  | scaleFactor_5: %f\n",scale_factor[3],scale_factor[4],scale_factor[5]);
    //mexPrintf("scaleFactor_6: %f | scaleFactor_7: %f  | scaleFactor_8: %f\n",scale_factor[6],scale_factor[7],scale_factor[8]);

	mxDestroyArray(mxGetField(outCell, activeSockNum, "socket"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "timestamp"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "csi_len"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "channel"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "payload_len"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "phyerr"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "noise"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "rate"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "chanBW"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "num_tones"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "nr"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "nc"));	
	mxDestroyArray(mxGetField(outCell, activeSockNum, "rssi"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "rssi_0"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "rssi_1"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "rssi_2"));	
	mxDestroyArray(mxGetField(outCell, activeSockNum, "csi"));
	mxDestroyArray(mxGetField(outCell, activeSockNum, "pkt_idx"));
    mxDestroyArray(mxGetField(outCell, activeSockNum, "MAC_idx"));
    
	mxSetField(outCell, activeSockNum, "socket", mxCreateDoubleScalar((double)fd));
	mxSetField(outCell, activeSockNum, "timestamp", timestamp);
	mxSetField(outCell, activeSockNum, "csi_len", mxCreateDoubleScalar((double)csi_len));
	mxSetField(outCell, activeSockNum, "channel", mxCreateDoubleScalar((double)channel));
	mxSetField(outCell, activeSockNum, "payload_len", mxCreateDoubleScalar((double)payload_len));
	mxSetField(outCell, activeSockNum, "phyerr", mxCreateDoubleScalar((double)phyerr));
	mxSetField(outCell, activeSockNum, "noise", mxCreateDoubleScalar((double)noise));
	mxSetField(outCell, activeSockNum, "rate", mxCreateDoubleScalar((double)rate));
	mxSetField(outCell, activeSockNum, "chanBW", mxCreateDoubleScalar((double)chanBW));
	mxSetField(outCell, activeSockNum, "num_tones", mxCreateDoubleScalar((double)num_tones));
	mxSetField(outCell, activeSockNum, "nr", mxCreateDoubleScalar((double)nr));
	mxSetField(outCell, activeSockNum, "nc", mxCreateDoubleScalar((double)nc));
	mxSetField(outCell, activeSockNum, "rssi", mxCreateDoubleScalar((double)rssi));
	mxSetField(outCell, activeSockNum, "rssi_0", mxCreateDoubleScalar((double)rssi_0));
	mxSetField(outCell, activeSockNum, "rssi_1", mxCreateDoubleScalar((double)rssi_1));
	mxSetField(outCell, activeSockNum, "rssi_2", mxCreateDoubleScalar((double)rssi_2));
	mxSetField(outCell, activeSockNum, "csi",csi);
    mxSetField(outCell, activeSockNum, "pkt_idx", mxCreateDoubleScalar((double)pkt_idx));
    mxSetField(outCell, activeSockNum, "MAC_idx", mxCreateDoubleScalar((double)MAC_idx));
}
void mexFunction(int nlhs, mxArray *plhs[],
			int nrhs, const mxArray *prhs[])
{
	const char* fieldnames[] = {"socket",
		"timestamp", 		
		"csi_len", 		
		"channel",
		"payload_len",
		"phyerr",
		"noise",
		"rate",
		"chanBW",
		"num_tones",
		"nr",
		"nc",
		"rssi",
		"rssi_0",
		"rssi_1",
		"rssi_2",
		"csi",
        "pkt_idx",
        "MAC_idx"};
        
	int *sock_vec;
	int *sock_vec_pr;
	int sock_max,sock_min,sock_cnt;
	int fd,client_sockfd;

	struct sockaddr_in client_addr;
	int addrlen = sizeof(struct sockaddr_in);
	int newSockFlag = 0;
	int activeSockNum = 0;
	int activeSockNum_all = 0;
		
	int recvlen = 0;
	int ret;
	int cnt,i,print_tmp;
	int CPUendian;

	mxArray *outCell;		/* The cell output matrix */
	
	char *cdata = (char *)0;
	MatVar mv;
	double timeout = -1;
	fd_set readfds,readfds_test,exceptfds;

	if(nrhs < 1) {
		mexPrintf("Must input a socket\n");
		return;
	}
	if(!mxIsNumeric(prhs[0])) {
		mexPrintf("First argument must be numeric.\n");
		return;
	}
	if(!mxIsNumeric(prhs[2])) {
		mexPrintf("2rd argument must be numeric.\n");
		return;
	}
	if(!mxIsNumeric(prhs[3])) {
		mexPrintf("3rd argument must be numeric.\n");
		return;
	}	
	if(nrhs > 4) {
		if(!mxIsNumeric(prhs[4])) {
			mexPrintf("4nd argument (timeout in s) must be numeric.\n");
			return;
		}
		timeout = mxGetScalar(prhs[4]);
	}

	sock_cnt = (int)mxGetScalar(prhs[0]);
	//printf("sock cnt is:%d\n",sock_cnt);
	
	sock_vec = (int *)mxGetData(prhs[1]);

//	for(i=0;i<sock_cnt;i++){ 
//		printf("Sock-%d in sock vector is: %d\n",i,sock_vec[i]);
//	}
	
	sock_min = (int)mxGetScalar(prhs[2]);
//	printf("sock min is:%d\n",sock_min);
	
	sock_max = (int)mxGetScalar(prhs[3]);
//	printf("sock max is:%d\n",sock_max);


	FD_ZERO(&readfds);
	FD_ZERO(&exceptfds);
	
	for(i=0;i<sock_cnt;i++){
// 		mexPrintf("i is: %d\n",i);
		FD_SET(sock_vec[i],&readfds);
		FD_SET(sock_vec[i],&exceptfds);
	}
	
	//printf("We are going to select\n");
	if(timeout < 0){
		//printf("Timeout negative\n");
		ret = select(sock_max+1,&readfds,NULL,&exceptfds,(struct timeval *)0);
		if(ret == -1){
			perror("select failed\n");
			exit(EXIT_FAILURE);
		}else if(ret == 0){
			//mexPrintf("No socket ready for read!\n");
			if(nlhs > 0){
				plhs[0]  = mxCreateNumericMatrix(sock_cnt,1,mxINT32_CLASS,mxREAL);
				sock_vec_pr = (int *)mxGetData(plhs[0]);
				for(i=0;i<sock_cnt;i++){
					sock_vec_pr[i] = sock_vec[i];
				}
			}  
			if(nlhs > 1){	
				plhs[1] = mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);         
				((int *)mxGetPr(plhs[1]))[0] = sock_cnt;
			}
			if(nlhs > 2){                                 
                outCell = mxCreateStructMatrix(0, 0, 19, fieldnames);
				plhs[2] = outCell;
            }
			return;
		}
	}else {
		struct timeval tv;
		tv.tv_sec = (int) timeout;
		tv.tv_usec = (int) (fmod(timeout,1.0)*1.0E6);
		
		//printf("Timeout is set: sec %d | usec %d\n",tv.tv_sec,tv.tv_usec);

		ret = select(sock_max+1,&readfds,NULL,&exceptfds,&tv);
		if(ret == -1){
			perror("select failed\n");
			exit(EXIT_FAILURE);
		}else if(ret == 0){
			//mexPrintf("No socket ready for read!\n");
			if(nlhs > 0){
				plhs[0]  = mxCreateNumericMatrix(sock_cnt,1,mxINT32_CLASS,mxREAL);
				sock_vec_pr = (int *)mxGetData(plhs[0]);
				for(i=0;i<sock_cnt;i++){
					sock_vec_pr[i] = sock_vec[i];
				}
			}  
			if(nlhs > 1){	
				plhs[1] = mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);         
				((int *)mxGetPr(plhs[1]))[0] = sock_cnt;
			}
			if(nlhs > 2){                                 
                outCell = mxCreateStructMatrix(0, 0, 19, fieldnames);
				plhs[2] = outCell;
            }
			return;
		}
	}

	for(fd=sock_min;fd<=sock_max;fd++){
		if(FD_ISSET(fd,&readfds)){
			if(fd != sock_vec[0]){
				activeSockNum_all += 1;
			}
		}
	}

	outCell = mxCreateStructMatrix(1,activeSockNum_all, 19, fieldnames);
	
	for(fd=sock_min;fd<=sock_max;fd++){
		if(FD_ISSET(fd,&readfds)){
			if(fd == sock_vec[0]){ 
				mexPrintf("we get connection from clients! Accept it!\n");
				// new client comes, accept the connection 
#if !defined(WIN32)
				client_sockfd = (int) accept(sock_vec[0],(struct sockaddr *)&client_addr,(socklen_t *)&addrlen);	
#else
				client_sockfd = (int) accept(sock_vec[0],(struct sockaddr *)&client_addr,(int *)&addrlen);	
#endif
				//mexPrintf("  Client_sockfd is: %d\n",client_sockfd);
				newSockFlag = 1;
			}
			else{
				// receive data from the socket
				// Receive the count first
				//mexPrintf("we get data from data from socket: %d\n",fd);
				activeSockNum += 1;
				cnt = 0;
				while (cnt < (int) sizeof(int)) {
					ret = ::recv(fd,((char *)&recvlen)+cnt,sizeof(int)-cnt,0);
					if(ret == -1) {
						perror("recv");
						plhs[0] = mxCreateNumericMatrix(0,0,mxDOUBLE_CLASS,mxREAL);
						if(nlhs > 1)
							plhs[1] = mxCreateDoubleScalar(-1.0f);
						return;
					}
					cnt += ret;
				}

				CPUendian = checkCPUendian();
     				if(CPUendian == 1){	
				//	mexPrintf("SWAP recvlen\n",ret);
    					unsigned char *tmp = (unsigned char *)&recvlen;
    					unsigned char t;
    					t = tmp[0];tmp[0] = tmp[3];tmp[3] = t;
    					t = tmp[1];tmp[1] = tmp[2];tmp[2] = t;
				}
				if(recvlen <= 0) {
					plhs[0] = mxCreateNumericMatrix(0,0,mxDOUBLE_CLASS,mxREAL);
					if(nlhs > 1)
						plhs[1] = mxCreateDoubleScalar(-1.0);
					return;
				}
				// Receive the array
				cdata = new char[recvlen];
				cnt = 0;
				while(cnt < recvlen) {
					ret = recv(fd,(cdata+cnt),recvlen-cnt,0);
					if(ret == -1) {
						delete[] cdata;
						perror("recv");
						plhs[0] = mxCreateNumericMatrix(0,0,mxDOUBLE_CLASS,mxREAL);
						if(nlhs > 1)
							plhs[1] = mxCreateDoubleScalar(-1.0);
						return;
					}
					cnt += ret;
				}

				if(activeSockNum <= activeSockNum_all){
					fill_csi_struct((unsigned char*)cdata,outCell,recvlen,activeSockNum-1,fd);
				}else{
					mexPrintf("Sock num unmatch!\n");
					break;
				}
				if(cdata) delete[] cdata;
			}
		}		
	}

	
	if(newSockFlag){ 			// we get new socket from client 
		sock_cnt += 1;			// update the socket number
		//mexPrintf("sock cnt is:%d\n",sock_cnt);
		plhs[0]  = mxCreateNumericMatrix(sock_cnt,1,mxINT32_CLASS,mxREAL);  // prepare the output
		sock_vec_pr = (int *)mxGetData(plhs[0]); 
		for(i=0;i<sock_cnt-1;i++){
			sock_vec_pr[i] = sock_vec[i];
		}
		//mexPrintf("sock_cnt is: %d | client_sockfd is:%d\n",sock_cnt,client_sockfd);
		sock_vec_pr[sock_cnt-1] = client_sockfd;   // adding the new socket into the socket vector
	}else{  				// no new socket coming, directly output all existing socket
		plhs[0]  = mxCreateNumericMatrix(sock_cnt,1,mxINT32_CLASS,mxREAL);
		sock_vec_pr = (int *)mxGetData(plhs[0]);
		for(i=0;i<sock_cnt;i++){
			sock_vec_pr[i] = sock_vec[i];
		}
	}

	plhs[1] = mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);        // output the socket number  
	((int *)mxGetPr(plhs[1]))[0] = sock_cnt;

	plhs[2] = outCell;  // output the CSI structure (socket is also included in the structure)
	
	return;
}
