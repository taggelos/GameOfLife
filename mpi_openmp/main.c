#include "header.h"

int main(int argc, char *argv[])
{
	int ntasks, taskid;
	int neighbors[4];

	MPI_Request reqs[8];
	MPI_Comm cartcomm; 

	// Initialize MPI

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

	// Check user input

	switch (ntasks)
	{
		case 1:
		case 4:
		case 9:
		case 16:
		case 25:
		case 36:
		case 49:
			break;
		default:
			if (taskid == MASTER)
				puts("!!! Choose a square number of workers between 1 and 49 !!!");
			MPI_Abort(MPI_COMM_WORLD, ERROR);
			exit(1);
	}

	int n = sqrt(ntasks);
	int subsize = NPROB/n;

	if (NPROB % n != 0)
	{
		if (taskid == MASTER)
			puts("!!! Choose a square number of workers multiple of NPROB !!!");
		
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	}

	int dimensions[2] = {n,n};
	int periods[2] = {1,1};
	MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 1, &cartcomm); // Create Cartesian Topology
	MPI_Comm_rank(cartcomm, &taskid);
	MPI_Cart_shift(cartcomm, 0, 1, &neighbors[UP], &neighbors[DOWN]);
	MPI_Cart_shift(cartcomm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);

	if (taskid == MASTER)
	{
		master(neighbors, cartcomm, ntasks, n, subsize);
	}
	else
	{
		Finalize* fin = worker(neighbors, cartcomm, subsize, taskid, n);
		finalize(fin);
	}

	MPI_Finalize();

	return 0;
}
