/*
 * File: serialCN.cpp
 * Author: Qian YANG
 * Date: Submitted 6/15/2014
 * Purpose:
 *		Implements serial version of Crank-Nicolson time evolution.
 *		Uses Eigen's dense PP LU solver and BiCGSTAB solver.
 *
 * Usage: ./serialCN ecut latconst solver
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h> // for pow
#include <stdlib.h>
#include <sys/time.h>
#include "CompCell.h"

int main(int argc, char **argv)
{
  // input parameters are number of mesh points in x and y directions
  if (argc < 4)
  {
    std::cerr << "Usage: " << argv[0] << " ecut latconst solver" << std::endl;
    return(1);
  }
  double ecut = atof(argv[1]);
  double latconst = atof(argv[2]);
  int solver = atoi(argv[3]); // 1: BiCGSTAB, else: LU

  // Set up time evolution
  double timestep = 1;
  double approxnumwv = 2;
  double totaltime = approxnumwv*13.3;
  int numsteps = ceil(totaltime/timestep);
  totaltime = numsteps*timestep;
	
  VectorXd dtlocs = VectorXd::LinSpaced(numsteps+1,0.25,0);
  double dstep = dtlocs(1)-dtlocs(0);
  std::cout << "Distance per timestep: " << dstep << std::endl;
  std::cout << "Time per timestep: " << timestep << std::endl;
  std::cout << "Total distance: " << 0.5*latconst << " Bohr" << std::endl;
  std::cout << "Total time: " << totaltime << std::endl;
		
  // Initialize computational cell; form initial Hamiltonian
  CompCell twohydrogen(ecut, latconst, dtlocs(0)*latconst);

  /***
      Crank-Nicolson time evolution
  ***/
  // std::cout << "Starting CN time evolution..." << std::endl;
  // /** Start Timer ***/
  // struct timeval start, end, prevend;
  // double dt;
  // gettimeofday(&start, NULL);
  // prevend = start;
	
  // for (int i=1; i<numsteps; i++)
  // {
  //   std::cout << "start iteration " << i << std::endl;
  //   twohydrogen.crank_nicolson_update(i, 1, timestep, dtlocs, solver);
		
  //   gettimeofday(&end, NULL);
  //   dt = ((end.tv_sec  - prevend.tv_sec) * 1000000u + end.tv_usec - prevend.tv_usec) / 1.e6;
  //   std::cout << "iteration " << i << " runtime (seconds): " << dt << std::endl;
  //   prevend = end;
  // }
	
  // // Subspace Diagonalization; first compute H at final timestep
  // Vector3d delta(0, 0, dtlocs(numsteps));
  // twohydrogen.compute_subdiag(1,delta);

  /*** End Timer ***/
  // gettimeofday(&end, NULL);
  // dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  // std::cout << "Serial time evolution (seconds): " << dt << std::endl;
	
  return 0;
}

