#include "header.h"

inline void master(int* neighbors, MPI_Comm cartcomm, int ntasks, int n, int subsize)
{
	int** arr = SeqAllocate(NPROB); // array for grid 

	printf("Starting mpi_life with %d worker tasks.\n", ntasks);

	printf("Grid size: %d Generations: %d\n", NPROB, GENERATION);
	printf("Initializing grid and writing in initial.dat  \n");
	inidat(NPROB, arr);
	

	prtdat(NPROB, arr, "initial.dat");

	// Take neighbors of workers

	int i;
	int start[2];
	int subsizes[2] = {subsize,subsize};
	int gridsizes[2] = {NPROB, NPROB};

	MPI_Datatype* grid = malloc(ntasks*sizeof(MPI_Datatype));
	
	//Create subarrays

	for (i=0; i<ntasks; i++)
	{
		MPI_Cart_coords(cartcomm, i, 2, start);
		start[0] *= subsize;
		start[1] *= subsize;
		MPI_Type_create_subarray(2, gridsizes, subsizes, start, MPI_ORDER_C, MPI_INT, &grid[i]);
		MPI_Type_commit(&grid[i]);
	}
	
	// Send to workers

	MPI_Request* reqs = malloc(2*ntasks*sizeof(MPI_Request));
	double time = MPI_Wtime();
	
	for (i=0; i<ntasks; i++)
	{
		MPI_Isend(&arr[0][0], 1, grid[i], i, BEGIN, cartcomm, &reqs[i]);
	}
	
	// Finalize
	
	Finalize* fin = worker(neighbors, cartcomm, subsize, MASTER, n);

	// Receive from worker

	for (i=0; i<ntasks; i++)
	{
		MPI_Irecv(&arr[0][0], 1, grid[i], i, DONE, cartcomm, &reqs[i+ntasks]);
	}

	// Wait all workers

	MPI_Waitall(2*ntasks, reqs, MPI_STATUSES_IGNORE);
	
	time = MPI_Wtime() - time;

	finalize(fin);

	
	printf("Finished after %f seconds\n", time);
	printf("Writing in final.dat \n");
	prtdat(NPROB, arr, "final.dat");

	// Frees
	for (i=0; i<ntasks; i++)
	{
		MPI_Type_free(&grid[i]);
	}
	free(grid);
	SeqFree(arr);
}
