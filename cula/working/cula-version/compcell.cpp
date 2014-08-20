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

#include <cula.h>
#include <cula_lapack.h>

#include "compcell.hpp"

using namespace Eigen;


void print_matrix( char* desc, int m, int n, double* a, int lda ) {
  int i, j;
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



  // For now, put SCF loop here
  for (int n = 0; n < nmax_iters; n++)
  {
    // Possible loop over k here -- for now, skip.

    //gettimeofday(&start, NULL);
    _set_H_kin();
    //gettimeofday(&end, NULL);
    //dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    //std::cout << "Time (sec) for _set_H_kin(): " << dt << std::endl;
    
    //gettimeofday(&start, NULL);
    _set_H_pot();
    // _set_H_ps(); Alternative to Hydrogen potential! 
    
    //gettimeofday(&end, NULL);
    //dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    //std::cout << "Time (sec) for _set_H_pot(): " << dt << std::endl;
    
    // NEW!!!
    _set_H_hartree();
    
    

    dt = _compute_eigs();
    std::cout << "Time (sec) for _compute_eigs(): " << dt << std::endl;
  }
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
	double G_G = pw.squaredNorm()/2;
	if (G_G < _ecut) {

	  num_plane_waves += 1;
	  plane_waves.push_back(pw(0));
	  plane_waves.push_back(pw(1));
	  plane_waves.push_back(pw(2));

	  _G2.push_back(G_G);
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


double CompCell::_set_H_ps()
{
  for (int i = 0; i < _num_plane_waves; i++)
  {
    for (int j = i+1; j < _num_plane_waves; j++)
    {
      Vector3d delta_G = _plane_waves.col(i) - _plane_waves.col(j);
      double G2 = delta_G.squaredNorm();
      if (G2 < _eps)
      {
	_form_factor = 

    }
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

void CompCell::_set_H_hartree()
{


}


void checkStatus(culaStatus status)
{
  char buf[256];
  if (!status)
    return;

  culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
  printf("%s\n", buf);

  culaShutdown();
  exit(EXIT_FAILURE);
}


void CompCell::_convert_H_to_culaDouble(const std::vector<double> &shiftedH, culaDouble *shiftedH_cula)
{
  for (int i = 0; i < _num_plane_waves; i++)
  {
    for (int j = 0; j < _num_plane_waves; j++)
    {
      culaDouble val = (culaDouble)shiftedH[i + _num_plane_waves*j];
      shiftedH_cula[i + _num_plane_waves*j] = val;
    }
  }
}


double CompCell::_compute_eigs()
{
  // Construct shifted H for faster computation
  std::vector<double> shiftedH(_num_plane_waves*_num_plane_waves);
  shiftedH = _H;
  culaDouble* shiftedH_cula = NULL;
  shiftedH_cula = (culaDouble*)malloc(_num_plane_waves*_num_plane_waves*sizeof(culaDouble));
  
  for (int i = 0; i < _num_plane_waves; i++)
    shiftedH[i + i*_num_plane_waves] -= (culaDouble)10000;

  _convert_H_to_culaDouble(shiftedH, &shiftedH_cula[0]);
  
  // Use CULA
  culaStatus status;

  char jobz = 'N';
  char uplo = 'U';
  int N = _num_plane_waves;
  int lwork = -1;
  int lda = N;

  status = culaInitialize();
  checkStatus(status);

  double start_time, end_time, cula_time;
  culaDouble w[N];

  struct timeval start, end;
  double dt;
  
  gettimeofday(&start, NULL);
  status = culaDsyev(jobz, uplo, N, shiftedH_cula, lda, w);
  checkStatus(status);
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

  for (int i = 0; i < _num_plane_waves; i++)
    w[i] += 10000;

  // print_matrix((char*)"Eigenvalues", 1, N, w, 1);

  return dt;
}
