#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

//Dimension of problem grid
#define NPROB 240     	 
#define GENERATION 100000	
//Size of threads should be multiple of warp's size(32)     
#define THREAD_NUMBER 64 	

//Standard inidat

void inidat(int nx, bool *u)
{
	int ix, iy;
	for (ix = 0; ix <= nx - 1; ix++)
	{
		for (iy = 0; iy <= nx - 1; iy++)
		{
			u[ix*NPROB+iy] = rand() % 2;
		}
	}
}

//Standard prtdat

void prtdat(int nx, bool* u, char *fnam)
{
	int ix, iy;
	FILE *fp;
	fp = fopen(fnam, "w");
	for (ix = 0; ix < nx; ix++)
	{
		for (iy = 0; iy < nx; iy++)
		{
			fprintf(fp, "%d", u[ix*NPROB+iy]);
			if (iy != nx - 1)
			{
				fprintf(fp, " ");
			}
			else
			{
				fprintf(fp, "\n");
			}
		}
	}
	fclose(fp);
}


__global__ void Update(bool *A, bool* B)
{
	//Take Indexes

	int i, j,temp,sum,down,up,left,right;
	temp = blockIdx.x * blockDim.x + threadIdx.x;
	i = temp / NPROB;
	j = temp % NPROB;


	//Check for errors and values of edges

	if (i < 0 || i > NPROB - 1 || j < 0 || j > NPROB - 1)
	{
		return;
	}

	down = (i != NPROB-1) ? i+1 : 0;
	up = (i != 0) ? i-1 : NPROB-1;
	left = (j != 0) ? j-1 : NPROB-1;
	right = (j != NPROB-1) ? j+1 : 0;

	sum = A[down*NPROB+j] + A[up*NPROB+j] + A[i*NPROB+left] + A[i*NPROB+right] + A[down*NPROB+left] + A[up*NPROB+left] + A[down*NPROB+right] + A[up*NPROB+right];
	
	//Calculate formula

	B[i*NPROB+j] = ((sum == 3) || (A[i*NPROB+j] == true && sum == 2));

	__syncthreads();
}

	


int main(int argc,char *argv[])
{
	bool *d_A,*d_B, *h_A;
	int it;
	
	//Creating two 1d arrays for cuda 

	cudaMalloc((void**)&d_A,(unsigned long)(NPROB*NPROB*sizeof(bool)));
	cudaMalloc((void**)&d_B,(unsigned long)(NPROB*NPROB*sizeof(bool)));

	//Creating h_A to initialiaze 

	h_A=(bool*)malloc(NPROB*NPROB*sizeof(bool));
	memset(h_A,0,NPROB*NPROB*sizeof(bool));
	
	//Transfering the h_A to device 

	printf("Grid size: %d Generations: %d\n", NPROB, GENERATION);
	printf("Initializing grid and writing in initial.dat  \n");

	inidat(NPROB,h_A);
	prtdat(NPROB, h_A, "initial.dat");
	cudaMemcpy(d_B,h_A,NPROB*NPROB*sizeof(bool),cudaMemcpyHostToDevice);
	cudaMemcpy(d_A,h_A,NPROB*NPROB*sizeof(bool),cudaMemcpyHostToDevice);	
	
	bool* handler;
	
	struct timeval  tv1, tv2;
	gettimeofday(&tv1, NULL);
	
	for (it = 1; it <= GENERATION; it++)
	{
		//Swapping between the two arrays
       	
		Update<<<NPROB*NPROB/THREAD_NUMBER+1,THREAD_NUMBER>>>(d_A,d_B);	

		cudaThreadSynchronize();

        handler = d_A;
		d_A = d_B;
		d_B = handler; 
   		
	}
			
	gettimeofday(&tv2, NULL);
	printf("Finished after %f seconds\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec));
	printf("Writing final.dat file \n");

	//Copy results back to host memory 

	cudaMemcpy(h_A,d_A, NPROB*NPROB*sizeof(bool), cudaMemcpyDeviceToHost) ;	

	//Print Results

	prtdat(NPROB, h_A, "final.dat");
	
	
	cudaFree(d_A);	
	cudaFree(d_B);
	
    free(h_A);
	
}
