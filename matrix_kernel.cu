#ifndef _MATRIX_KERNEL_H_
#define _MATRIX_KERNEL_H_

//**===Copies the path to the min_path in order to save it===**//
__device__ void copy_path(int *path_min,int *path,int N){
	int i;
	for(i=0;i<N;i++){
		path_min[i]=path[i];
	}
}


//**=== Moves one spot down(or to the next city) in the matrix===**//
//**=== Returns 0 when a city circle is complete. 1 when it found a city but no circle is complete yet. ===**//
//**===-1 when when called after completing a circle===**//
__device__ int move_down(float *sD,int N,int *pos,int *path,float *current){
	int i,j,found;
	if(*pos < N-2){
		for(i=1;i<N;i++){
			found=0;
			for(j=0;j<N;j++){
				if(path[j]==i){
					found=1;
					break;
				}
			}
			if(found==0){
				path[*pos+1]=i;
				if(*pos!=-1)*current += sD[path[*pos]*N+path[*pos+1]];
				else *current = sD[path[*pos+1]];
				*pos+=1;
				break;
			}
		}
		return 1;
	}else if(*pos==N-2){
		*current += sD[path[*pos]*N];
		path[N-1]=0;
		*pos+=1;
		return 0;
	}else{
		return -1;
	}
};

//**=== Deletes the last path entry and subtracts the corresponding distance from the current sum===**//
//**=== Returns 0 on success,1 when you reach the minimum depth for one thread and -1 on failure===**//
__device__ int move_up(float *sD,int N,int *pos,int depth,int *path,float *current){
	if(*pos==-1){
		return -1;
	}
	if(*pos==(depth))return 1;
	if(*pos!=0) *current-=sD[path[*pos]*N+path[*pos-1]];
	else *current = 0;
	path[*pos]=-1;
	*pos-=1;
	return 0;
};

//**=== If we have the path B -> C -> D this function finds the next alternative to D, like B -> c -> E===**//
//**=== if it cant find an alternative it returns 1 else 0 on error -1 ===**//
__device__ int move_next(float *sD,int pos,int N,int *path,float *current){
	int i,found,j;
	if(pos==-1){
		return -1;
	}
	for(i=path[pos];i<N;i++){
		found=0;
		for(j=0;j<N;j++){
			if(path[j]==i){
				found=1;
				break;
			}
		}
		if(found==0){
			if(pos==0)*current -= sD[path[pos]*N+0];
			else *current -= sD[path[pos]*N+path[pos-1]];
			path[pos]=i;
			if(pos==0)*current += sD[path[pos]*N+0];
			else *current += sD[path[pos]*N+path[pos-1]];
			break;
		}
	}
	if(found==1)return 1;
	else return 0;
};


//**=== Kernel function called from the cpu program ===**//
__global__ void kernel(float *R,float *D,int N,size_t pitchD,int depth){

	//**=== Variables initialization ===**//
	int jj=threadIdx.x%N;
	int ii=(int)threadIdx.x / N;
	
	int min_path[21];
	int path[21];
	int pos=-1;
	float min=0;
	float current=0;
	int counter=1;
	__shared__ float sD[20*20]; // storaze of the Distances array in the shared memory
	__shared__ float buff[256]; //buffer used to calculate the smallest min in the block
	
	//**=== Copy the Distances array to the shared memory===**//
	int x_step = (int)blockDim.x / N;
	int y_step = x_step;
	while(ii < N){
		while(jj < N){
			float *rD=(float *)((char *)D + pitchD*ii);
			sD[ii*N+jj] = rD[jj];
			jj+=y_step;
		}
		jj=threadIdx.x%N;
		ii+=x_step;
	}
	__syncthreads();
	
	
	//**=== Initialzie the min_path and  path arrays===**//
	for(ii=0;ii<21;ii++){
		min_path[ii]=-1;
		path[ii]=-1;
	}
	
	
	//**=== Chose a path according to the thread ID until you reach the specified depth===**//
	int kk;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	for(ii=N-1;ii > (N - 1 - depth);ii--){
		int temp = (index%(ii));
		move_down(sD,N,&pos,path,&current);
		for(kk=0;kk<temp;kk++){
			move_next(sD,pos,N,path,&current);
		}
	}
	counter = path[0]*1000+path[1]*100+path[2]*10+path[3]*1;
	//**Acquire the first combination and set it as the min**//
	while(move_down(sD,N,&pos,path,&current)!=-1){
		;
	}
	min=current;
	copy_path(min_path,path,N);
	//**===The main loop that find all the possible combinations that are assigned to this thread===**//
	while(move_up(sD,N,&pos,depth,path,&current)==0){
		while(move_next(sD,pos,N,path,&current)==0){
			while(move_down(sD,N,&pos,path,&current)==1){
				;
			}
			counter++;
			if(current < min){
				min=current;
				copy_path(min_path,path,N);
			}
		}
	}
	
	//**=== Here we calculate the min of mins :P===**//
	buff[threadIdx.x]=min;
	__syncthreads();
	
	int size = blockDim.x;
	while(size!=1){
		if(threadIdx.x < size/2){
			if(buff[threadIdx.x+(size/2)]<buff[threadIdx.x]){
				buff[threadIdx.x]= buff[threadIdx.x+(size/2)];
			}
			size=size/2;
		}else{
			size=1;
		}
		__syncthreads(); //sync the thread to save the changes to the shared memory
	}
	__syncthreads();

	//**=== The thread with id 0 copies the result back to the global memory===**//
	if(threadIdx.x==0){
		R[blockIdx.x] = buff[0];
	}
};
#endif