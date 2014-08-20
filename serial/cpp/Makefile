INC_EIGEN = -I /home/rehnd/src/eigen/

serial: main.cpp compcell.cpp
	g++ -o main main.cpp compcell.cpp -O3 $(INC_EIGEN)

clean:
	rm -f main *~
