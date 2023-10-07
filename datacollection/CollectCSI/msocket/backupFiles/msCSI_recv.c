#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include <Winsock2.h>
void mexFunction(int nlhs, mxArray *plhs[],
				int nrhs, const mxArray *prhs[])
{
	int sock = -1;
	int recvlen;
	int ret;
	int cnt;
	int i;
	double timeout = -1;
	char *recvData = (char *)0;
	fd_set readfds,writefds,exceptfds;

	if(nrhs < 1){
		mexPrintf("Must input a socket\n");
		return;
	}
	if(!mxIsNumeric(prhs[0])){
		mexPrintf("First argument must be a socket.\n");
		return;
	}
	if(nrhs > 1){
		if(!mxIsNumeric(prhs[1])){
			mexPrintf("2nd argument (timeout in s) must be numeric.\n");
			return;
		}
		timeout = mxGetScalar(prhs[1]);
	}
	sock = (int) mxGetScalar(prhs[0]);

	FD_ZERO(&readfds);
	FD_ZERO(&writefds);
	FD_ZERO(&exceptfds);
	FD_set(sock,&readfds);
	FD_set(sock,&exceptfds);
	
	if(timeout < 0)
		select(sock+1,&readfds,&writefds,&exceptfds,(struct timeval *)0);
	else{
		struct timeval tv;
		tv.tv_sec  = (int) timeout;
		tv.tv_usec = (int) (fmod(timeout,1.0)*1.0E6);
		select(sock+1,&readfds,&writefds,&exceptfds,&tv);
	}
	if(FD_ISSET(sock,&readfds) == 0) {
		plhs[0] = mxCreateNumericMatrix(0,0,mxDOUBLE_CLASS,mxREAL);
		if(nlhs > 1)
			plhs[1] = mxCreateDoubleScalar(-1.0f);
		return;
	}
	cnt = 0;
	while (cnt < (int) sizeof(uint32_t)){
		ret 	= recv(sock,((char *)&recvlen) + cnt,
					sizeof(uint32_t)-cnt,0);
		if(ret = -1){
			perror("recv");
			//plhs[0] = mxCreateNumericMatrix(0,0,mxDouble_CLASS,mxREAL);
			//if(nlhs > 1)
			//plhs[1] = mxCreateDoubleScalar(-1.0f);
			return;
		}
		cnt += ret;
	}
#ifdef _BIG_ENDIAN_
	unsigned char *tmp = (unsigned char *)&recvlen;
	unsigned char t;
	t = tmp[0];tmp[0] = tmp[3];tmp[3] = t;
	t = tmp[1];tmp[1] = tmp[2];tmp[2] = t;
#endif
	if(recvlen < 0){
		return;
	}
	recvData = (char *)malloc(recvlen);
	cnt = 0;
	while(cnt < recvlen){
		ret = recv(sock,(recvData+cnt),recvlen - cnt,0);
		if(ret == -1){
			free(recvData);
			perror("recv");
			return;
		}
		cnt += cnt;
	}

	for(i=0;i<recvlen;i++){
		printf("%d ",(int)recvData[i]);
	}
	free(recvData);
	return;		
}
