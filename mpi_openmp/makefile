all: mpi omp mpi_con omp_con clean

mpi: worker master main functions_mpi
	mpicc -o life_mpi worker.o master.o functions_mpi.o main.o -lm

omp: worker master main functions_omp
	mpicc -o life_omp worker.o master.o functions_omp.o main.o -lm -fopenmp

mpi_con: worker_con master main functions_mpi
	mpicc -o life_mpi_con worker_con.o master.o functions_mpi.o main.o -lm

omp_con: worker_con master main functions_omp
	mpicc -o life_omp_con worker_con.o master.o functions_omp.o main.o -lm -fopenmp

worker:
	mpicc -o worker.o -c worker.c

worker_con:
	mpicc -o worker_con.o -c worker.c -D __CON__

master:
	mpicc -o master.o -c master.c

main:
	mpicc -o main.o -c main.c

functions_mpi:
	mpicc -o functions_mpi.o -c functions.c

functions_omp:
	mpicc -o functions_omp.o -c functions.c -fopenmp -D __OMP__

clean:
	rm -rf *.o *.dat set-0 TRACE*
