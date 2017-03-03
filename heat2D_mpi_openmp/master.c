#include "header.h"

inline void master(int* nbrs, MPI_Comm cartcomm, int numtasks, int n, int subsize)
{
	int** u = SeqAllocate(NPROB); /* array for grid */

	printf("Starting mpi_heat2D with %d worker tasks.\n", numtasks);

	/* Initialize grid */
	printf("Grid size: %d Time steps= %d\n", NPROB, GENERATION);
	printf("Initializing grid and writing initial.dat file...\n");
	inidat(NPROB, u);
	/* REMOVE */
	//int z,zz;
	//for (z = 0; z<NPROB; z++)
	//for (zz = 0; zz<NPROB; zz++)
	//u[z][zz]=z*NPROB+zz;
	/**/
	prtdat(NPROB, u, "initial.dat");

	// Take neighbors of workers
	int i;
	int starts[2];
	int subsizes[2] = {subsize,subsize};
	int bigsizes[2] = {NPROB, NPROB};
	MPI_Datatype* grid = malloc(numtasks*sizeof(MPI_Datatype));
	puts("AXNEEEEEEEEEEEEEEEEEEEEEEE11111...");
	for (i=0; i<numtasks; i++)
	{
		MPI_Cart_coords(cartcomm, i, 2, starts);
		starts[0] *= subsize;
		starts[1] *= subsize;
		MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &grid[i]);
		MPI_Type_commit(&grid[i]);
	}
	puts("AXNEEEEEEEEEEEEEEEEEEEEEE222222221...");
	// Send to workers
	MPI_Request* reqs = malloc(2*numtasks*sizeof(MPI_Request));
	double time = MPI_Wtime();
	for (i=0; i<numtasks; i++)
	{
		MPI_Isend(&u[0][0], 1, grid[i], i, BEGIN, cartcomm, &reqs[i]);
	}
	puts("AXNEEEEEEEEEEEEEEEEEEEEEEE3333333333...");
	Finalize* fin = worker(nbrs, cartcomm, subsize, MASTER);

	// Receive from worker
	for (i=0; i<numtasks; i++)
	{
		MPI_Irecv(&u[0][0], 1, grid[i], i, DONE, cartcomm, &reqs[i+numtasks]);
	}

	// Wait all workers
	MPI_Waitall(2*numtasks, reqs, MPI_STATUSES_IGNORE);
	time = MPI_Wtime() - time;

	finalize(fin);

	/* Write final output, call X graph and finalize MPI */
	printf("Finished after %f seconds\n", time);
	printf("Writing final.dat file \n");
	prtdat(NPROB, u, "final.dat");

	// Free structures
	for (i=0; i<numtasks; i++)
	{
		MPI_Type_free(&grid[i]);
	}
	free(grid);
	SeqFree(u);
}
