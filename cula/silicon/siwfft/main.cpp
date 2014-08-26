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
  if (argc != 2)
  {
    std::cerr << "Usage:\n\t" << argv[0] << " <E_cut>" << std::endl;
    return(1);
  }

  double ecut = atof(argv[1]);
  double latconst = 10.26;
  int nk = 1; // For now, will be rewritten in cell.cpp)

  // Initialize computational cell, run scf
  cell silicon(ecut, latconst, nk);
  silicon._scf();

  return 0;
}
