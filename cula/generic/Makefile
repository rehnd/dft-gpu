INC_EIGEN = -I /home/rehnd/src/eigen/

CULA_INC=-I${CULA_INC_PATH}
CULA_LIBS=-L${CULA_LIB_PATH_64}

LIBS=-lcula_lapack -lcublas -lcudart -liomp5

cula: main.cpp compcell.cpp
	nvcc -o culamain main.cpp compcell.cpp -O3 $(INC_EIGEN) $(CULA_INC) $(CULA_LIBS) ${LIBS}

clean:
	rm -f main culamain *~