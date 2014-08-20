#include <iostream>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h> // for pow
#include <stdlib.h>
#include <sys/time.h>

#include "cell.hpp"

int main(int argc, char* argv[])
{
  // input parameters are number of mesh points in x and y directions
  if (argc < 4)
  {
    std::cerr << "Usage:\n\t" << argv[0] << " <E_cut> <Lat_const> <n_k>" << std::endl;
    return(1);
  }

  double ecut = atof(argv[1]);
  double latconst = atof(argv[2]);
  int nk = atoi(argv[3]); // Number of k points to use

  // Initialize computational cell, run scf
  cell silicon(ecut, latconst, nk);
  silicon._scf();

  return 0;
}
