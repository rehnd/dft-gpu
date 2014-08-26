FFTW_INC=-I${FFTW_INC_PATH}
FFTW_LIB=-L${FFTW_LIB_PATH}
LIBS=-lfftw3

main: test.c
	g++ -o test test.c -O3 -g ${FFTW_INC} ${FFTW_LIB} ${LIBS}

clean:
	rm -rf main *~
