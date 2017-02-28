#include "header.h"

int main(int argc, char *argv[])
{
	int numtasks, taskid;
	int nbrs[4];

	MPI_Request reqs[8];
	MPI_Comm cartcomm; // required variable

	// Initialize MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

	// Check number of proccess from user input
	switch (numtasks)
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
			{
				puts("ERROR: the number of workers must be between 1 and 49 and must be a square number.");
				puts("Quitting...");
			}
			MPI_Abort(MPI_COMM_WORLD, ERROR);
			exit(1);
	}

	int n = sqrt(numtasks);
	int subsize = NPROB/n;

	if (NPROB%n != 0)
	{
		if (taskid == MASTER)
		{
			puts("ERROR: square root of number of workers must be multiple of NPROB");
			puts("Quitting...");
		}
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	}

	// Create Cartesian Topology
	int dims[2] = {n,n};
	int periods[2] = {1,1};
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cartcomm);
	MPI_Comm_rank(cartcomm, &taskid);
	MPI_Cart_shift(cartcomm, 0, 1, &nbrs[UP], &nbrs[DOWN]);
	MPI_Cart_shift(cartcomm, 1, 1, &nbrs[LEFT], &nbrs[RIGHT]);
	//MPI_Cart_shift(cartcomm, 0, 1, &nbrs[UPLEFT], &nbrs[DOWNLEFT]);
	//MPI_Cart_shift(cartcomm, 0, 1, &nbrs[UPRIGHT], &nbrs[DOWNRIGHT]);
	


	if (taskid == MASTER)
	{
		master(nbrs, cartcomm, numtasks, n, subsize);
	}
	else
	{
		Finalize* fin = worker(nbrs, cartcomm, subsize, taskid);
		finalize(fin);
	}

	MPI_Finalize();

	return 0;
}
