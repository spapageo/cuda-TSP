/*  
 * 4h ergasia Parallila k Katanemimena Systimata 2010-11
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <cuda/cuda_runtime.h>
#include <cutil.h>

#define BLOCK_SIZE 128

#define PI 3.14159265


float *result;
float *Dist;
float *dev_D=0; //device pointer to the distances matrix
float *dev_R=0; //device pointer to the results array
size_t pitchD;


#include "matrix_kernel.cu"

void print(float *A,int size);

float calc_min(float *R,int blocks);

int move_down_c(float *sD,int N,int *pos,int *path,float *current);

int move_up_c(float *sD,int N,int *pos,int depth,int *path,float *current);

int move_next_c(float *sD,int pos,int N,int *path,float *current);

int init_cuda_mem(int N,int blocks);

int free_cuda_mem();

int main(int argv,char **argc) {

    if (argv!=2) {
        printf("Usage: %s N\n",argc[0]);
        exit(EXIT_FAILURE);
    }


    srand(time(NULL));

    //** Variable declaration**//
    unsigned int i,j,r = 1;
    unsigned int threads=0,depth=0;
    unsigned int combinations=0;
    unsigned int N = atoi(argc[1]);
    long long int fN=1;
    float cor_min;


    struct timeval first, second, lapsed;
    struct timezone tzp;
    cudaError_t err;
    float Pos[N][2];
    dim3 block;
    block.y = 1;
    block.z = 1;
    dim3 grid;
    grid.y = 1;
    grid.z = 1;



    /*=== Calculate factorial !(N-1)===*/
    for (i=2;i<N;i++) {
        fN = fN * (long long int)i;
    }



    //**=== Create a proper N-agonal ===**//
    Dist=(float *)malloc(N*N*sizeof(float));

    //printf("Positions array:\n");
    for (i = 0; i < N; i++) {  //upologismos twn thesewn panw sto kanoniko polugwno
        Pos[i][0]=r * cos(2 * PI * i / N);
        Pos[i][1]=r * sin(2 * PI * i / N);
        //printf("%f %f\n",Pos[i][0], Pos[i][1]);
    }

    for (i=0;i<N;i++) { //upologismos pinaka  apostasewn
        Dist[i*N + i]=0;
        for (j=i+1;j<N;j++) {
            Dist[i*N + j]= sqrt(((Pos[i][0]-Pos[j][0])*(Pos[i][0]-Pos[j][0]))+((Pos[i][1]-Pos[j][1])*(Pos[i][1]-Pos[j][1])));
            Dist[j*N + i]= Dist[i*N+j];
        }
    }

    cor_min=N*2*r*sin(PI/N);

    //print(Dist,N);






    //**=== Calculate proper depth and number of threads to use. ===**//
    unsigned int THREADS_MAX=5041, mm=1, x=1;

    while (x<=THREADS_MAX && mm<N) {
        x*=N-mm;
        mm++;
    }
    threads=x/(N-mm+1);
    depth=mm-2;
    combinations = fN/(long long int)threads;
    if (combinations==1)depth++;
    
    if(N==12){
       threads = 7920;
       combinations = 5040;
       depth = 4;
    } else if(N==13){
       threads = 11880;
       combinations = 40320;
       depth = 4;
    }
	    
    printf("Number of threads actually created: %d\n",threads);
    printf("Depth = %d\n",depth);
    printf("Combinations per thread %d\n",combinations);

    //**=== Calculate the grid and block size ===**//
    i=1;
    j=threads;
    while ((j%2)==0) {
        i*=2;
        j=j/2;
    }
    block.x = i;
    grid.x = j;
    printf("Block size %d, grid size %d\n",block.x,grid.x);
    
    /*=== Initialize the device memory and the result buffer===*/
    init_cuda_mem(N,grid.x);
    result=(float *)malloc(grid.x*sizeof(float));


    gettimeofday(&first, &tzp);

    kernel<<<grid,block>>>(dev_R,dev_D,N,pitchD,depth);

    cudaThreadSynchronize();

    CUDA_SAFE_CALL(cudaMemcpy((void *)(result),(void *)dev_R,grid.x*sizeof(float),cudaMemcpyDeviceToHost));

    float min = calc_min(result,grid.x);

    gettimeofday(&second, &tzp);

    err = cudaGetLastError();

//     //**==============Testing Area========================**//
//     int kk;
//     int index=101;
//     int pos=-1;
//     int counter=1;
//     int path[30];
//     float current=0;
// 
//     for (i=0;i<N;i++) {
//         path[i]=-1;
//     }
// 
//     for (i=N-1;i > (N - 1 - depth);i--) {
//         int temp = (index%(i));
//         move_down_c(Dist,N,&pos,path,&current);
//         for (kk=0;kk<temp;kk++) {
//             move_next_c(Dist,pos,N,path,&current);
//         }
//     }
//     printf("The current path is:\n");
//     for (i=0;i<N;i++) {
//         printf("%d ",path[i]);
//     }
//     
//     while (move_down_c(Dist,N,&pos,path,&current)!=-1) {
//         ;
//     }
//     min=current;
// 
//     printf("\nThe starting path is path is:\n");
//     for (i=0;i<N;i++) {
//         printf("%d ",path[i]);
//     }
//     
//     while (move_up_c(Dist,N,&pos,depth,path,&current)==0) {
//         while (move_next_c(Dist,pos,N,path,&current)==0) {
//             while (move_down_c(Dist,N,&pos,path,&current)==1) {
//                 ;
//             }
//             counter++;
// 	    printf("Counter increased to %d\n",counter);
//             if (current < min) {
//                 min=current;
//             }
//         }
//     }
// 
// 
//     //counter = path[0]*1000+path[1]*100+path[2]*10+path[3]*1;
//     printf("\nThe counter is %d\n",counter);
// 
//     //printf("The results are:\n");
//     //for (i=0;i<grid.x;i++) {
//     //    printf("%f ",result[i]);
//     //}

    printf("\n");

    if (first.tv_usec>second.tv_usec) {
        second.tv_usec += 1000000;
        second.tv_sec--;
    }

    lapsed.tv_usec = second.tv_usec - first.tv_usec;
    lapsed.tv_sec = second.tv_sec - first.tv_sec;
    printf("--------------------------\nGPU time\n");
    printf("Time elapsed: %d.%06dsec\n", (int)lapsed.tv_sec, (int)lapsed.tv_usec);


    printf("Kernel finished with: %s\n",cudaGetErrorString(err));

    printf("Minimum distance covering all the cities is %f\n",min);
    printf("The correct minimum distance is %f\n",cor_min);

    //free(Pos);
    free(Dist);
    free_cuda_mem();

    return EXIT_SUCCESS;
}



int init_cuda_mem(int N,int blocks) {

    CUDA_SAFE_CALL(cudaMallocPitch((void **)&dev_D,&pitchD,N * sizeof(float),N));

    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_R,blocks*sizeof(float)));

    CUDA_SAFE_CALL(cudaMemcpy2D((void *)(dev_D),pitchD,(void *)Dist,(N)*sizeof(float),(N)*sizeof(float),N,cudaMemcpyHostToDevice));

    printf("Initialization complete!\n");

    return 0;

}


int free_cuda_mem() {
    cudaFree((void *)dev_D);
    cudaFree((void *)dev_R);
    return 0;
}



void print(float *A,int size) {
    int i,j;
    printf("\n Distances Array: \n");

    for (i=0; i<size; i++) {
        for (j=0;j<size;j++)
            printf("%.3f ", A[i*size+j]);
        printf("\n");
    }
    printf("\n");
}


float calc_min(float *R,int blocks) {
    int i;
    float min=RAND_MAX;
    for (i=0;i<blocks;i++) {
        if (R[i]<min) min=R[i];
    }
    return min;
}

int move_down_c(float *sD,int N,int *pos,int *path,float *current) {
    int i,j,found;
    if (*pos < N-2) {
	int old = path[*pos];
        for (i=1;i<N;i++) {
            found=0;
            for (j=0;j<N;j++) {
                if (path[j]==i) {
                    found=1;
                    break;
                }
            }
            if (found==0) {
                path[*pos+1]=i;
                if (*pos!=-1)*current += sD[path[*pos]*N+path[*pos+1]];
                else *current += sD[path[*pos+1]];
                *pos+=1;
                break;
            }
        }
        printf("Moving down %d to %d\n",old,path[(*pos)]);
        return 1;
    } else if (*pos==N-2) {
	printf("Moving down %d to %d\n",path[(*pos)],0);
        *current += sD[path[*pos]*N];
        path[N-1]=0;
        *pos+=1;
        return 0;
    } else {
        return -1;
    }
};

int move_up_c(float *sD,int N,int *pos,int depth,int *path,float *current) {
    if (*pos==-1) {
        return -1;
    }
    if (*pos==(depth))return 1;
    if (*pos!=0) *current-=sD[path[*pos]*N+path[*pos-1]];
    else *current = 0;
    printf("Moving up %d to %d\n",path[*pos],path[(*pos)-1]);
    path[*pos]=-1;
    *pos-=1;
    return 0;
};


int move_next_c(float *sD,int pos,int N,int *path,float *current) {
    int i,found,j;
    if (pos==-1) {
        return -1;
    }
    int old = path[pos]; 
    for (i=path[pos];i<N;i++) {
        found=0;
        for (j=0;j<N;j++) {
            if (path[j]==i) {
                found=1;
                break;
            }
        }
        if (found==0) {
            if (pos==0)*current -= sD[path[pos]*N+0];
            else *current -= sD[path[pos]*N+path[pos-1]];
            path[pos]=i;
            if (pos==0)*current += sD[path[pos]*N+0];
            else *current += sD[path[pos]*N+path[pos-1]];
            break;
        }
    }
    if (found==1)return 1;
    else{
      printf("Moving sideways %d to %d\n",old,path[pos]);
      return 0;
    } 
};
