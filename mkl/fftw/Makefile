INC_EIGEN = -I ~/src/eigen/

MKL_INC  = -I/home/rehnd/src/mkl
FFTW_INC = -I${FFTW_INC_PATH}
FFTW_LIB = -L${FFTW_LIB_PATH}

LIBS = -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -ldl -lm -lpthread -lfftw3 -lm

main: main.cpp cell.cpp tools.cpp
	nvcc -o main main.cpp cell.cpp tools.cpp -O3 -g -arch=sm_20 ${INC_EIGEN} ${FFTW_INC} ${FFTW_LIB} ${MKL_INC} ${LIBS}

clean:
	rm -rf main *~
