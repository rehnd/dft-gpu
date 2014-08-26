#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/LU>
#include <sys/time.h>

#include "mkl_lapacke.h"
#include "compcell.hpp"

using namespace Eigen;

void print_matrix( char* desc, MKL_INT m, MKL_INT n, double* a, MKL_INT lda ) {
  MKL_INT i, j;
  printf("\n%s: \n", desc);
  for( i = 0; i < m; i++ ) {
    for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
    printf( "\n" );
  }
}

// set up computational cell with constructor
CompCell::CompCell(double ecut, double latconst, double initdist) : _ecut(ecut), _latconst(latconst)
{
  // define lattice vectors
  _a1 << 0.5*latconst, 0.5*latconst, 0;
  _a2 << 0, 0.5*latconst, 0.5*latconst;
  _a3 << 0.5*latconst, 0, 0.5*latconst;

  // compute reciprocal lattice vectors
  _vol = _a1.dot(_a2.cross(_a3));
  _b1 = (2*M_PI/_vol)*_a2.cross(_a3);
  _b2 = (2*M_PI/_vol)*_a3.cross(_a1);
  _b3 = (2*M_PI/_vol)*_a1.cross(_a2);

  // timer variables
  struct timeval start, end;
  double dt;

  // compute plane waves
  std::vector<double> plane_waves; // temporary container

  gettimeofday(&start, NULL);
  _get_plane_waves(plane_waves);
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  std::cout << "Time (sec) for _get_plane_waves: " << dt << std::endl;

  _plane_waves = Map<MatrixXd>(plane_waves.data(),3,_num_plane_waves);

  _H.resize(_num_plane_waves*_num_plane_waves, 0);

  gettimeofday(&start, NULL);
  _set_H_kin();
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  std::cout << "Time (sec) for _set_H_kin(): " << dt << std::endl;

  gettimeofday(&start, NULL);
  _set_H_pot();
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  std::cout << "Time (sec) for _set_H_pot(): " << dt << std::endl;

  gettimeofday(&start, NULL);
  _compute_eigs();
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  std::cout << "Time (sec) for _compute_eigs(): " << dt << std::endl;

}


void CompCell::_get_plane_waves(std::vector<double>& plane_waves)
{
  /*** find the integral linear combinations of reciprocal lattice vecs
       with electron free particle energy less than ecut ***/

  std::vector<int> maxceil;
  maxceil.push_back(ceil(sqrt(_ecut)*sqrt(2)/_b1.norm()));
  maxceil.push_back(ceil(sqrt(_ecut)*sqrt(2)/_b2.norm()));
  maxceil.push_back(ceil(sqrt(_ecut)*sqrt(2)/_b3.norm()));
  int max = *(std::max_element(maxceil.begin(), maxceil.end()));

  int num_plane_waves = 0;

  for (int i=-max; i<=max; i++) {
    for (int j=-max; j<=max; j++) {
      for (int k=-max; k<=max; k++) {

	Vector3d pw = i*_b1 + j*_b2 + k*_b3;
	if (pw.squaredNorm()/2 < _ecut) {

	  num_plane_waves += 1;
	  plane_waves.push_back(pw(0));
	  plane_waves.push_back(pw(1));
	  plane_waves.push_back(pw(2));
	}
      }
    }
  }
  _num_plane_waves = num_plane_waves;
  std::cout << "Num plane waves: " << _num_plane_waves << std::endl;
}


void CompCell::_set_H_kin()
{
  for (int i = 0; i < _num_plane_waves; i++)
  {
    Vector3d k_G = _plane_waves.col(i);
    _H[i + _num_plane_waves*i] = k_G.squaredNorm()/2;
  }
}

void CompCell::_set_H_pot()
{
  for (int i = 0; i < _num_plane_waves; i++)
  {
    for (int j = i+1; j < _num_plane_waves; j++)
    {
      Vector3d delta_G = _plane_waves.col(i) - _plane_waves.col(j);
      double H_orig_pot = -4*M_PI/_vol/delta_G.squaredNorm();

      _H[i + _num_plane_waves*j] = H_orig_pot;
    }
  }
}


void CompCell::_compute_eigs()
{
  // Construct shifted H for faster computation
  std::vector<double> shiftedH(_num_plane_waves*_num_plane_waves);
  shiftedH = _H;
  for (int i = 0; i < _num_plane_waves; i++)
    shiftedH[i + i*_num_plane_waves] -= 10000;


  // Use MKL
  MKL_INT n = _num_plane_waves;
  MKL_INT lda = n;
  MKL_INT info;

  double w[n];

  info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, &shiftedH[0], lda, w);
  if (info > 0)
  {
    printf("Algorithm failed to compute eigenvalues\n");
    exit(1);
  }

  for (int i = 0; i < _num_plane_waves; i++)
    w[i] += 10000;

  // print_matrix((char*)"Eigenvalues", 1, n, w, 1);

}
