INC_EIGEN = -I /home/rehnd/src/eigen/

MKL_INC=-I /home/rehnd/src/mkl
MKL_LIBS=-lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -ldl -lm -lpthread

mkl: main.cpp compcell.cpp
	g++ -o main main.cpp compcell.cpp -O3 $(INC_EIGEN) $(MKL_INC) $(MKL_LIBS)

clean:
	rm -f main *~
