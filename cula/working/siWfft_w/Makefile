INC_EIGEN = -I ~/src/eigen/

CULA_INC = -I${CULA_INC_PATH}
CULA_LIB = -L${CULA_LIB_PATH_64}
FFTW_INC = -I${FFTW_INC_PATH}
FFTW_LIB = -L${FFTW_LIB_PATH}

LIBS=-lcula_lapack -lcublas -lcudart -liomp5 -lcuda -lcufft -lfftw3 -lm

main: main.cpp cell.cpp tools.cpp
	nvcc -o main main.cpp cell.cpp tools.cpp -O3 -g ${INC_EIGEN} ${CULA_INC} ${CULA_LIB} ${FFTW_INC} ${FFTW_LIB} ${LIBS}

clean:
	rm -rf main *~
