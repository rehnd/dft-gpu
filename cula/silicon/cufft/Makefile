INC_EIGEN = -I ~/src/eigen/

CULA_INC = -I${CULA_INC_PATH}
CULA_LIBS = -L${CULA_LIB_PATH_64}

LIBS=-lcula_lapack -lcublas -lcudart -liomp5 -lcufft

main: main.cpp cell.cpp tools.cpp
	nvcc -o main main.cpp cell.cpp tools.cpp -O3 -g -arch=sm_20 ${INC_EIGEN} ${CULA_INC} ${CULA_LIBS} ${LIBS}

clean:
	rm -rf main *~
