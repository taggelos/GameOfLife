life_game_cuda: life_game_cuda.cu
	nvcc $^ -o $@	-arch=sm_20	
 
clean:
	\rm -f life_game_cuda	

rebuild: clean life_game_cuda
