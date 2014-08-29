INC_EIGEN = -I ~/src/eigen/

CULA_INC = -I${CULA_INC_PATH}
CULA_LIBS = -L${CULA_LIB_PATH_64}

LIBS=-lcula_lapack -lcudart -liomp5

main: main.cpp cell.cpp
	nvcc -o main main.cpp cell.cpp -O3 -arch=sm_20 ${INC_EIGEN} ${CULA_INC} ${CULA_LIBS} ${LIBS}

clean:
	rm -rf main *~
