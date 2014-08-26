#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h> // for pow
#include <stdlib.h>
#include <sys/time.h>
#include "compcell.hpp"

int main(int argc, char* argv[])
{
  // input parameters are number of mesh points in x and y directions
  if (argc < 4)
  {
    std::cerr << "Usage:\n\t" << argv[0] << " <E_cut> <Lat_const> <solver>" << std::endl;
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
  // std::cout << "Distance per timestep: " << dstep << std::endl;
  // std::cout << "Time per timestep: " << timestep << std::endl;
  // std::cout << "Total distance: " << 0.5*latconst << " Bohr" << std::endl;
  // std::cout << "Total time: " << totaltime << std::endl;

  // Initialize computational cell; form initial Hamiltonian
  CompCell twohydrogen(ecut, latconst, dtlocs(0)*latconst);

  return 0;
}
