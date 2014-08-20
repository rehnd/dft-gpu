CULA_INC = -I${CULA_INC_PATH}
CULA_LIBS = -L${CULA_LIB_PATH_64}

LIBS=-lcula_lapack -lcublas -lcudart -liomp5 -lcufft

main: fft.cu
	nvcc -o main fft.cu -O3 ${INC_EIGEN} ${CULA_INC} ${CULA_LIBS} ${LIBS}

clean:
	rm -rf main *~


