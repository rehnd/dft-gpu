#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <sstream>
#include <map>
#include <complex>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/LU>
#include <sys/time.h>

#include <cula.h>
#include <cula_lapack.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cufft.h>

#include "cell.hpp"

using namespace Eigen;
using std::string;

void print_matrix( char* desc, int m, int n, double* a, int lda ) {
  int i, j;
  printf("\n%s: \n", desc);
  for( i = 0; i < m; i++ ) {
    for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
    printf( "\n" );
  }
}


// set up computational cell with constructor
cell::cell(double ecut, double latconst, int nk) : _ecut(ecut), _latconst(latconst)
{
  _nk = 1;
  _k.resize(3, _nk);
  _npw_perk.resize(_nk);
  _wk.resize(1,1);

  double _tau1 = 0.125, _tau2 = 0.125, _tau3 = 0.125;

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

  _k(0,0) = 0.6223; _k(1,0) = 0.2953; _k(2,0) = 0.0;

  _get_plane_waves();
  _G = Map<MatrixXd>(_G.data(),3,_npw);  /// !!!!!!! Very careful here! not sure what this does.

  _get_SG();
  _count_nk();

}


std::string miller(int i0, int i1, int i2)
{
  std::stringstream ss;
  ss << i0 << " " << i1 << " " << i2;
  return ss.str();
}


void cell::_get_plane_waves()
{
  // Timing:
  struct timeval start, end;   double dt;   gettimeofday(&start, NULL);

  std::vector<int> maxceil;
  maxceil.push_back(ceil(sqrt(_ecut)*sqrt(2)/_b1.norm()));
  maxceil.push_back(ceil(sqrt(_ecut)*sqrt(2)/_b2.norm()));
  maxceil.push_back(ceil(sqrt(_ecut)*sqrt(2)/_b3.norm()));
  int max = *(std::max_element(maxceil.begin(), maxceil.end()));

  int npw = 0;

  for (int i=-max; i<=max; i++) {
    for (int j=-max; j<=max; j++) {
      for (int k=-max; k<=max; k++) {

	Vector3d pw = i*_b1 + j*_b2 + k*_b3;
	double G_G = pw.squaredNorm()/2;

	if (G_G < _ecut)   npw++;
      }
    }
  }

  _G2.resize(npw);
  _G.resize(3, npw);
  _mill.resize(3,npw);

  int ng = 0;
  for (int i = -max; i <= max; i++) {
    for (int j = -max; j <= max; j++) {
      for (int k = -max; k <= max; k++) {

	Vector3d pw = i*_b1 + j*_b2 + k*_b3;
	double G_G = pw.squaredNorm()/2;

	if (G_G < _ecut) {
	  _G.col(ng) << pw;
	  _G2[ng] = G_G;
	  _mill.col(ng) << i, j, k;
	  ng++;
	}
      }
    }
  }

  if (ng != npw)
  {
    printf("Error in _get_plane_waves: ng != npw\n");
    exit(1);
  }
  _npw = npw;
  std::cout << "Num plane waves: " << _npw << std::endl;

  // Now fill Miller Indices into indg
  _nm0 = _G.rowwise().maxCoeff()(0);
  _nm1 = _G.rowwise().maxCoeff()(1);
  _nm2 = _G.rowwise().maxCoeff()(2);

  for (int i = 0; i < _npw; i++)
  {
    string str = miller( _mill(0,i), _mill(1,i), _mill(2,i) );
    _indg[str] = i;
  }
  // While we're here, set the Real-Space grid dimensions:
  _nr0 = 2 * _nm0 + 2;
  _nr1 = 2 * _nm1 + 2;
  _nr2 = 2 * _nm2 + 2;

  // Timing:
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  std::cout << "Time (sec) for _get_plane_waves: " << dt << std::endl;
}


void cell::_get_SG(void)
{
  // Calculates the geometrical structure factors S(G)

  _SG.resize(_npw);

  for (int i = 0; i < _npw; i++)
    _SG[i] = 2.0*cos( 2*M_PI*(_G(0,i)*_tau0 + _G(1,i)*_tau1 + _G(2,i)*_tau2) ) / _vol ;
}


void cell::_count_nk(void)
{
  for (int k = 0; k < _nk; k++)
  {
    // Count plane waves such that (\hbar^2/2m) (k+G)^2 < E_cut
    _npw_perk[k] = 0;

      for (int n = 0; n < _npw; n++)
      {
      	VectorXd kg = _k.col(k) + _G.col(n);
      	double kg2 = kg.squaredNorm();
	if (kg2 <= _ecut)
	  _npw_perk[k]++;
      }
    printf("Number of plane waves for k-vector %d = %d\n", k, _npw_perk[k]);
  }

  _npw_max = *max_element(_npw_perk.begin(), _npw_perk.end());
  printf("Max number of k+G values across all k = %d\n", _npw_max);

  _igk.resize(_npw_max, _nk);

  for (int k = 0; k < _nk; k++)
  {
    int nn = 0;
    for (int n = 0; n < _npw; n++)
    {
      VectorXd kg = _k.col(k) + _G.col(n);
      double kg2  = kg.squaredNorm();

      if (kg2 <= _ecut)
      {
  	_igk(nn,k) = n;
  	nn++;
      }
    }
    if (nn != _npw_perk[k])
    {
      printf("Mismatch in number of plane waves: nn (= %d) != _npw_perk[k] (=%d)\n", nn, _npw_perk[k]);
      exit(1);
    }
  }
}


double cell::_form_factor(double G2)
{
  _e2 = 2.;
  _eps = 1.e-8;
  // Pseudopotential (Applebaum-Hamann) parameters:
  double v1 = 3.042, v2 = -1.372, alpha = 0.6102, zv = 4.;

  double form_factor;
  if (G2 < _eps)
  {
    // G = 0: divergent Hartree and pseudopotential terms cancel. Calculate what is left analytically
    form_factor = _e2 * M_PI * zv / alpha + _e2 * pow(M_PI/alpha, 1.5) * (v1 + 1.5 * v2 / alpha);
  }
  else
  {
    form_factor = _e2 * exp(-G2/4./alpha) * (-4*M_PI*zv/G2 + pow(M_PI/alpha, 1.5) *
					     (v1 + v2/alpha * (1.5 - G2/4/alpha) ) );
  }
  return form_factor;
}


void cell::_fillH(int k)
{
  _H.resize(_npw_perk[k] * _npw_perk[k]);

  for (int i = 0; i < _npw_perk[k]; i++)
  {
    int ik = _igk(i, k);
    Vector3d kg = _k.col(k) + _G.col(ik);

    for (int j = i; j < _npw_perk[k]; j++)
    {
      int jk = _igk(j, k);

      int n1 = _mill(0, ik) - _mill(0, jk);
      int n2 = _mill(1, ik) - _mill(1, jk);
      int n3 = _mill(2, ik) - _mill(2, jk);

      string str = miller(n1,n2,n3);
      int ng = _indg[str];

      double vsg = _form_factor( _G2[ng] );

      if (i == j)
	_H[i + j*_npw_perk[k]] = kg.squaredNorm() + vsg * _SG[ng] + _v[ng];
      else
	_H[i + j*_npw_perk[k]] = vsg * _SG[ng] + _v[ng];
    }
  }
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


double cell::_diagH(int k)
{
  int npw = _npw_perk[k];
  culaDouble* culaH = NULL;
  culaH = (culaDouble*)malloc(npw*npw*sizeof(culaDouble));

  // First need to convert H to culaDouble
  for (int i = 0; i < npw; i++)
  {
    for (int j = 0; j < npw; j++)
    {
      culaDouble val = (culaDouble)_H[i + npw*j];
      culaH[i + npw*j] = val;
      if (j == i)
	culaH[i + npw*i] -= 10000;
    }
  }

  culaStatus status;
  char jobz = 'V';
  char uplo = 'U';
  int N = npw;
  int lwork = -1;
  int lda = N;

  status = culaInitialize();
  checkStatus(status);

  double start_time, end_time, cula_time, dt;
  struct timeval start, end;

  culaDouble w[N];

  gettimeofday(&start, NULL);
  status = culaDsyev(jobz, uplo, N, culaH, lda, w);
  checkStatus(status);
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

  for (int i = 0; i < npw; i++)
    w[i] += 10000;

  _eigvecs.resize(npw*_nbands);
  _eigvals.resize(_nbands);

  for (int i = 0; i < npw; i++)
  {
    for (int j = 0; j < _nbands; j++)
    {
      double val = (double) culaH[i + j*_nbands];
      _eigvecs[i + j*_nbands] = val;
      _eigvals[j] = w[j];
    }
  }

  // printf("Eigenvalues:   ");
  // for (int i = 0; i < _eigvals.size(); i++)
  //   std::cout << _eigvals[i] << " ";
  // printf("      ");

  return dt;
}


void cell::_calcRho(int k)
{
  int npw = _npw_perk[k];
 
  /* Calculate \rho in reciprocal space:
     \rho(G) = {1\over \Omega} \sum_{G'}\psi^*(G'-G)\psi(G') */
  for (int i = 0; i < npw; i++)
  {
    int ik = _igk(i, k);
    for (int j = 0; j < npw; j++)
    {
      int jk = _igk(j, k);

      int n1 = _mill(0, ik) - _mill(0, jk);
      int n2 = _mill(1, ik) - _mill(1, jk);
      int n3 = _mill(2, ik) - _mill(2, jk);

      string str = miller(n1,n2,n3);
      int ng = _indg[str];

      for (int nb = 0; nb < _nbands; nb++)
      {
	_rhoout[ng] += _wk[k]*_eigvecs[i*_nbands + nb]*_eigvecs[j*_nbands + nb];
	// Would need conjugate for complex Hamiltonians
      }
    }
  }
}


void cell::_sumCharge(int k)
{

  int memsize = sizeof(cufftDoubleComplex)*_nr0*_nr1*_nr2;
  cufftDoubleComplex* h_aux = (cufftDoubleComplex*)malloc(memsize);
  // h_aux.resize(_nr0*_nr1*_nr2);
  printf("Made it\n");
  cufftDoubleComplex *d_aux;
  cufftHandle plan;
  cufftType CUFFT_C2C;
  int nrank = 3;

  int npw = _npw_perk[k];

  printf("Made it\n");

  for (int nb = 0; nb < _nbands; nb++)
  {
    for (int n0 = 0; n0 < _nr0; n0++)
    {
      for (int n1 = 0; n1 < _nr1; n1++)
      {
	for (int n2 = 0; n2 < _nr2; n2++)
	{
	  for (int i = 0; i < npw; i++)
	  {
	    int ik = _igk(i, k);
	    int m0 = _mill(0, ik);
	    if (m0 < 0)
	      m0 += _nr1;
	    int m1 = _mill(1, ik);
	    if (m1 < 0)
	      m1 += _nr1;
	    int m2 = _mill(2, ik);
	    if (m2 < 0)
	      m2 += _nr2;

	    // Miller indices run from -n/2 to n/2
	    // m0, m1, m2 run from 0 to _nr1, _nr2, _nr3
	    // with negative values refolded so they lie
	    // in the "far side of the cell" in G space

	    // std::cout << "m0: " << m0 << " m1: " << m1 << " m2: " << m2 << std::endl;

	    h_aux[m0 + m1*_nr1 + m2*_nr1*_nr2].x = _eigvecs[i*_nbands + nb];
	    h_aux[m0 + m1*_nr1 + m2*_nr1*_nr2].y = 0;
	    // d_aux[m0][m1][m2] = _eigvecs[i*_nbands + nb];

	  }
	}
      }
    }
    
    printf("Made it\n");
    cudaMalloc((void**)&d_aux, memsize);
    if (cudaGetLastError() != cudaSuccess)
    {
      fprintf(stderr, "Cuda error: Failed to allocate\n");
      return;
    }

    cudaMemcpy(&d_aux[0], &h_aux[0], memsize, cudaMemcpyHostToDevice);

    if (cufftPlan3d(&plan, _nr0, _nr1, _nr2, CUFFT_Z2Z) != CUFFT_SUCCESS)
    {
      fprintf(stderr, "CUFFT error: Plan creation failed\n");
      return;
    }

    if (cufftExecZ2Z(plan, &d_aux[0], &d_aux[0], CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
      fprintf(stderr, "CUFFT error: ExecZ2Z failed\n");
      return;
    }

    if (cudaDeviceSynchronize() != cudaSuccess)
    {
      fprintf(stderr, "Cuda error: Failed to synchronize\n");
      return;
    }
    
    cudaMemcpy(&h_aux[0], &d_aux[0], memsize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < npw; i++)
    {
      // Factor of 2 for spin degeneracy. 1/_vol comes from def of plane waves
      _rhoout[i] += 2*_wk[k]*std::abs(pow(h_aux[i].x, 2) + pow(h_aux[i].y, 2))/_vol;
    }

  }

  cufftDestroy(plan);
  cudaFree(d_aux);

}


void cell::_scf(void)
{
  // Timing:
  struct timeval start, end;   double dt;   gettimeofday(&start, NULL);

  _nbands = 8;
  _nelec = 8;
  _max_iter = 4;
  _alpha = 0.5; // Charge mixing parameter
  _threshold = 1.e-6; // Convergence threshold

  _rhoin.resize(_npw, 0.);
  _rhoout.resize(_npw, 0.);
  _v.resize(_npw, 0.);

  // Set \rho_in(G=0) to the correct value.
  string origin = miller(0,0,0);
  _rhoin[_indg[origin]] = _nelec/_vol;

  for (int iter = 0; iter < _max_iter; iter++)
  {
    _rhoout.resize(_npw, 0.);

    for (int k = 0; k < _nk; k++)
    {
      struct timeval start1, end1; double dt1; gettimeofday(&start1, NULL);
      _fillH(k);
      gettimeofday(&end1, NULL);
      dt1 = ((end1.tv_sec  - start1.tv_sec) * 1000000u + end1.tv_usec - start1.tv_usec) / 1.e6;
      std::cout << "Time (sec) for _fillH: " << dt1 << std::endl;

      gettimeofday(&start1, NULL);
      double dt = _diagH(k);
      // printf("Iteration %d, Diagonalization time = %g sec      ", iter, dt);
      gettimeofday(&end1, NULL);
      dt1 = ((end1.tv_sec  - start1.tv_sec) * 1000000u + end1.tv_usec - start1.tv_usec) / 1.e6;
      std::cout << "Time (sec) for _diagH: " << dt1 << std::endl;

      gettimeofday(&start1, NULL);
      _sumCharge(k);
      gettimeofday(&end1, NULL);
      dt1 = ((end1.tv_sec  - start1.tv_sec) * 1000000u + end1.tv_usec - start1.tv_usec) / 1.e6;
      std::cout << "Time (sec) for _calcRho: " << dt1 << std::endl;
    }

    struct timeval start1, end1; double dt1; gettimeofday(&start1, NULL);
    // Account for factor 2 (spin degeneracy) in rhoout:
    // for (int i = 0; i < _npw; i++)
    //   _rhoout[i] = 2. * _rhoout[i] / _vol;

    // Charge mixing

    double charge = 0;
    for (int i = 0; i < _npw; i++)
      charge += std::abs(_rhoout[i]);
    //charge *= _vol/(_nr0*_nr1*_nr2);
    std::cout << "Charge: " << charge << std::endl;

    double drho2 = 0.;
    for (int i = 0; i < _npw; i++)
      drho2 += pow(_rhoout[i] - _rhoin[i], 2);

    if ( sqrt(drho2) < _threshold)
    {
      printf("Convergence threshold %g reached\n", _threshold);
      break;
    }
    else 
      printf("Delta rho = %g\n", sqrt(drho2));
    
    for (int i = 0; i < _npw; i++)
      _rhoin[i] = _alpha * _rhoin[i] + (1.-_alpha)*_rhoout[i];

    // New charge is now in rhoout. Calculate new potential hartree term in G space
    for (int ng = 0; ng < _npw; ng++)
    {
      if (_G2[ng] > _eps)
	_v[ng] = 4*M_PI*_e2*_rhoin[ng]/_G2[ng];
      else
	_v[ng] = 0.;
    }
    gettimeofday(&end1, NULL);
    dt1 = ((end1.tv_sec  - start1.tv_sec) * 1000000u + end1.tv_usec - start1.tv_usec) / 1.e6;
    std::cout << "Time (sec) for all charge mixing stuff: " << dt1 << std::endl;

  }


  // Timing:
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  std::cout << "Time (sec) for _scf: " << dt << std::endl;

}
