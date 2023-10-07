////////////////////////////////////////////////////////////
//
// Name:   msrecv.cpp
//
// Author: Steven Michael (smichael@ll.mit.edu)
//
// Date:   5/19/06
//
// Description:
//
//    This is part of the "msocket" suite of TCP/IP 
//    funcitons for MATLAB.  It is a wrapper for the
//    "recv" socket function call. The data send is a serialized
//    MALTAB variable in a format described by matvar.cpp
//
// Copyright (c) 2006 MIT Lincoln Laboratory
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, 
// Boston, MA  02110-1301  USA
//
////////////////////////////////////////////////////////////

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
// void fill_csi_matrix(unsigned char *local_h, int nr, int nc, int num_tones);{
//     
// }
void fill_csi_struct(unsigned char *buf_addr, mxArray *outCell,int cnt){
    int real, imag;
    int h_data, h_idx, i, print_tmp;
	int bitmask, current_data;
    int bits_left, nc_idx, nr_idx;
	
	int k, idx;
	
	int csi_len    = ((buf_addr[8] << 8) & 0xff00) | (buf_addr[9] & 0x00ff);
	int channel 	= ((buf_addr[10] << 8) & 0xff00) | (buf_addr[11] & 0x00ff);
	int buf_len 	= ((buf_addr[cnt-2] << 8) & 0xff00) | (buf_addr[cnt-1] & 0x00ff);
	
	int payload_len = ((buf_addr[csi_st_len] << 8) & 0xff00) | ((buf_addr[csi_st_len + 1]) & 0x00ff);
	
	int phyerr  = buf_addr[12];
	int noise 	= buf_addr[13];
	int rate    = buf_addr[14];
	int chanBW 	= buf_addr[15];
	int num_tones 	= buf_addr[16];
	int nr      = buf_addr[17];
	int nc      = buf_addr[18];

	int rssi 	= buf_addr[19];
	int rssi_0 	= buf_addr[20];
	int rssi_1 	= buf_addr[21];
	int rssi_2 	= buf_addr[22];

	//unsigned char *local_h  = buf_addr + csi_st_len;
	unsigned char *local_h  = &buf_addr[25];

/*   	for(i=0;i<(int)csi_len;i++){
		print_tmp = local_h[i] & 0xFF;
		printf("0x%02x ",print_tmp);
	}
	printf("\n");
*/	

    const mwSize dims[] =   {nr, nc, num_tones};
    
	mxArray *csi  	= mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX);
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
			}
		}
	}
	mxDestroyArray(mxGetField(outCell, 0, "csi_len"));
	mxDestroyArray(mxGetField(outCell, 0, "channel"));
	mxDestroyArray(mxGetField(outCell, 0, "payload_len"));
	mxDestroyArray(mxGetField(outCell, 0, "phyerr"));
	mxDestroyArray(mxGetField(outCell, 0, "noise"));
	mxDestroyArray(mxGetField(outCell, 0, "rate"));
	mxDestroyArray(mxGetField(outCell, 0, "chanBW"));
	mxDestroyArray(mxGetField(outCell, 0, "num_tones"));
	mxDestroyArray(mxGetField(outCell, 0, "nr"));
	mxDestroyArray(mxGetField(outCell, 0, "nc"));	
	mxDestroyArray(mxGetField(outCell, 0, "rssi"));
	mxDestroyArray(mxGetField(outCell, 0, "rssi_0"));
	mxDestroyArray(mxGetField(outCell, 0, "rssi_1"));
	mxDestroyArray(mxGetField(outCell, 0, "rssi_2"));	
	mxDestroyArray(mxGetField(outCell, 0, "csi"));
	
	mxSetField(outCell, 0, "csi_len", mxCreateDoubleScalar((double)csi_len));
	mxSetField(outCell, 0, "channel", mxCreateDoubleScalar((double)channel));
	mxSetField(outCell, 0, "payload_len", mxCreateDoubleScalar((double)payload_len));
	mxSetField(outCell, 0, "phyerr", mxCreateDoubleScalar((double)phyerr));
	mxSetField(outCell, 0, "noise", mxCreateDoubleScalar((double)noise));
	mxSetField(outCell, 0, "rate", mxCreateDoubleScalar((double)rate));
	mxSetField(outCell, 0, "chanBW", mxCreateDoubleScalar((double)chanBW));
	mxSetField(outCell, 0, "num_tones", mxCreateDoubleScalar((double)num_tones));
	mxSetField(outCell, 0, "nr", mxCreateDoubleScalar((double)nr));
	mxSetField(outCell, 0, "nc", mxCreateDoubleScalar((double)nc));
	mxSetField(outCell, 0, "rssi", mxCreateDoubleScalar((double)rssi));
	mxSetField(outCell, 0, "rssi_0", mxCreateDoubleScalar((double)rssi_0));
	mxSetField(outCell, 0, "rssi_1", mxCreateDoubleScalar((double)rssi_1));
	mxSetField(outCell, 0, "rssi_2", mxCreateDoubleScalar((double)rssi_2));
	mxSetField(outCell, 0, "csi",csi);
}
void mexFunction(int nlhs, mxArray *plhs[],
			int nrhs, const mxArray *prhs[])
{
	const char* fieldnames[] = {"csi_len", 		
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
		"csi"};
	int sock = -1;
	int recvlen = 0;
	int ret;
	int cnt,i,print_tmp;
	int CPUendian;

	mxArray *outCell;		/* The cellular output */

	char *cdata = (char *)0;
	MatVar mv;
	double timeout = -1;
	fd_set readfds,writefds,exceptfds;

	if(nrhs < 1) {
		mexPrintf("Must input a socket\n");
		return;
	}
	if(!mxIsNumeric(prhs[0])) {
		mexPrintf("First argument must be a socket.\n");
		return;
	}
	if(nrhs > 1) {
		if(!mxIsNumeric(prhs[1])) {
			mexPrintf("2nd argument (timeout in s) must be numeric.\n");
			return;
		}
		timeout = mxGetScalar(prhs[1]);
	}

	sock = (int)mxGetScalar(prhs[0]);

	FD_ZERO(&readfds);
	FD_ZERO(&writefds);
	FD_ZERO(&exceptfds);
	FD_SET(sock,&readfds);
	FD_SET(sock,&exceptfds);

	if(timeout < 0){
		select(sock+1,&readfds,&writefds,&exceptfds,(struct timeval *)0);
	}else {
		struct timeval tv;
		tv.tv_sec = (int) timeout;
		tv.tv_usec = (int) (fmod(timeout,1.0)*1.0E6);
		select(sock+1,&readfds,&writefds,&exceptfds,&tv);
	}
	
	if(FD_ISSET(sock,&readfds)==0) {
		mexPrintf("Not ready to read!\n");
		plhs[0] = mxCreateNumericMatrix(0,0,mxDOUBLE_CLASS,mxREAL);
		if(nlhs > 1)
			plhs[1] = mxCreateDoubleScalar(-1.0f);
		return;
	}
	// Receive the count
	cnt = 0;
	while (cnt < (int) sizeof(int)) {
		ret = ::recv(sock,((char *)&recvlen)+cnt,sizeof(int)-cnt,0);
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
//	mexPrintf("Received length is:%d\n",recvlen);
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
		ret = recv(sock,(cdata+cnt),recvlen-cnt,0);
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
//	mexPrintf("Received %d bytes\n",cnt);

/*     	for(i=0;i<recvlen;i++){
		print_tmp = cdata[i] & 0xFF;
		printf("%d ",print_tmp);
	}
	printf("\n");
*/	
	outCell = mxCreateStructMatrix(1, 1, 15, fieldnames);
	fill_csi_struct((unsigned char*)cdata,outCell,recvlen);
	plhs[0] = outCell;
	if(cdata) delete[] cdata;
	return;
}
