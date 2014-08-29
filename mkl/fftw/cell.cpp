#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <sstream>
#include <map>
#include <sys/time.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/LU>

#include "mkl_lapacke.h"
#include <cufft.h>
#include <fftw3.h>

#include "cell.hpp"
#include "Array3d.hpp"
#include "IndG.hpp"
#include "tools.hpp"

using namespace Eigen;
using std::string;


cell::cell(double ecut, double latconst, int nk) : _ecut(ecut), _latconst(latconst)
{
  _nk = 1; // Number of k points to be used
  _k.resize(3, _nk);
  _npw_perk.resize(_nk); // Number of plane waves per k point
  _wk.resize(1,1); // Weights of each k point (initialize to 1)

  /* Below: Components of $\tau$ vector. Note: for now, leaving out
     a_0 which should multiply each component. Really should change
     this later so that S(G) = 2*cos(G\cdot \tau) with correct units */
  _tau0 = _latconst*0.125; _tau1 = _latconst*0.125; _tau2 = _latconst*0.125; 

  // Define lattice vectors
  _a1 << 0.5*latconst, 0.5*latconst, 0;
  _a2 << 0.5*latconst, 0, 0.5*latconst;
  _a3 << 0, 0.5*latconst, 0.5*latconst;

  // Compute reciprocal lattice vectors
  _vol = std::abs(_a1.dot(_a2.cross(_a3))); // _vol < 0 -> use abs()
  _b1 = -(2*M_PI/_vol)*_a2.cross(_a3);
  _b2 = -(2*M_PI/_vol)*_a3.cross(_a1);
  _b3 = -(2*M_PI/_vol)*_a1.cross(_a2);

  // k point used is a mean-value point
  _k(0,0) = 2*M_PI*0.6223/_latconst; 
  _k(1,0) = 2*M_PI*0.2953/_latconst;
  _k(2,0) = 0.0;

  _get_plane_waves();
  _get_SG();
  _count_nk();
}


void cell::_get_plane_waves()
{
  // Timing:
  struct timeval start, end;   double dt;   gettimeofday(&start, NULL);

  _nm0 = (int) round(sqrt( 4.*_ecut )/ (2*M_PI) *  sqrt(_a1.squaredNorm()) + 0.5);
  _nm1 = (int) round(sqrt( 4.*_ecut )/ (2*M_PI) *  sqrt(_a2.squaredNorm()) + 0.5);
  _nm2 = (int) round(sqrt( 4.*_ecut )/ (2*M_PI) *  sqrt(_a3.squaredNorm()) + 0.5);

  int npw = 0;
  // Calculate max # of plane waves based on E_{cut}
  for (int i = -_nm0; i <= _nm0; i++)
  {
    for (int j = -_nm1; j <= _nm1; j++)
    {
      for (int k = -_nm2; k <= _nm2; k++)
      {
	Vector3d pw = i*_b1 + j*_b2 + k*_b3;
	double G_G = pw.squaredNorm();

	if (G_G <= 4.*_ecut)   npw++;
      }
    }
  }

  _G2.resize(npw);
  _G.resize(3, npw);
  _mill.resize(3,npw);

  int ng = 0;
  for (int i = -_nm0; i <= _nm0; i++)
  {
    for (int j = -_nm1; j <= _nm1; j++)
    {
      for (int k = -_nm2; k <= _nm2; k++)
      {
	Vector3d pw = i*_b1 + j*_b2 + k*_b3;
	double G_G = pw.squaredNorm();

	if (G_G <= 4.*_ecut)
	{
	  _G(0, ng) = pw(0); _G(1,ng) = pw(1); _G(2,ng) = pw(2);
	  _G2[ng] = G_G;
	  _mill(0,ng) = i; _mill(1,ng) = j; _mill(2,ng) = k;
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
  printf("Number of plane waves: %d\n", _npw);

  // Fill Miller Indices into indg:
  _nm0 = _mill.rowwise().maxCoeff()(0);
  _nm1 = _mill.rowwise().maxCoeff()(1);
  _nm2 = _mill.rowwise().maxCoeff()(2);
  _indg.Initialize(_nm0, _nm1, _nm2);

  for (int i = 0; i < _npw; i++)
  {
    // string str = miller( _mill(0,i), _mill(1,i), _mill(2,i) );
    // _indg[str] = i;
    _indg(_mill(0,i), _mill(1,i), _mill(2,i)) = i;
  }

  // While we're here, set the Real-Space grid dimensions:
  _nr0 = 2 * _nm0 + 2;
  _nr1 = 2 * _nm1 + 2;
  _nr2 = 2 * _nm2 + 2;

  printf("Real-space grid dimensions n_x, n_y, n_z = %d, %d, %d\n\n", _nr0, _nr1, _nr2);

  // Timing:
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  printf("Time (sec) for _get_plane_waves: %g\n", dt);
}


void cell::_get_SG(void)
{
  // Calculates the geometrical structure factors S(G)
  _SG.resize(_npw, 0);
  for (int i = 0; i < _npw; i++)
    _SG[i] = 2.0*cos( _G(0,i)*_tau0 + _G(1,i)*_tau1 + _G(2,i)*_tau2 ) / _vol;
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
  char igkdesc[] = "igk";
}


double cell::_form_factor(double G2)
{
  _e2 = 2.; // Conversion factor from Rydberg to Hartree
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
  int npw = _npw_perk[k];
  _H.resize(npw*npw, 0.);

  for (int i = 0; i < npw; i++)
  {
    int ik = _igk(i, k); // May get more caching if switch i, k
    Vector3d kg = _k.col(k) + _G.col(ik);

    for (int j = i; j < npw; j++)
    {
      int jk = _igk(j, k);

      int n1 = _mill(0, ik) - _mill(0, jk);
      int n2 = _mill(1, ik) - _mill(1, jk);
      int n3 = _mill(2, ik) - _mill(2, jk);

      int ng = _indg(n1, n2, n3);
      
      double vsg = _form_factor( _G2[ng] );

      if (i == j)
	_H[i*npw + j] = kg.squaredNorm() + vsg * _SG[ng] + _vg[ng];
      else
	_H[i*npw + j] = vsg * _SG[ng] + _vg[ng];
    }
  }
}


double cell::_diagH(int k)
{
  int npw = _npw_perk[k];

  char jobz = 'V';
  char uplo = 'L'; // _H is upper triangular, let MKL use 'L' to keep same as cula code
  MKL_INT n = npw;
  MKL_INT lda = n;
  MKL_INT info;
  double w[n];

  double start_time, end_time, cula_time, dt;
  struct timeval start, end;

  gettimeofday(&start, NULL);
  info = LAPACKE_dsyev(LAPACK_COL_MAJOR, jobz, uplo, n, &_H[0], lda, w);
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

  // for (int i = 0; i < npw; i++)
  //   w[i] += 10000;

  _eigvecs.resize(npw*_nbands,0);
  _eigvals.resize(_nbands);

  for (int i = 0; i < _nbands; i++)
  {
    for (int j = 0; j < npw; j++)
    {
      double val = (double) _H[i*npw + j]; // eigenvectors are stored in col-major
      _eigvecs[j*_nbands + i] = val; // Save eigenvectors in row order
      _eigvals[i] = w[i];
    }
  }

  char evals[] = "Eigenvalues";
  // print_matrix(evals, 1, _eigvals.size(), &_eigvals[0]);

  return dt;
}


void cell::_sumCharge(int k)
{
  int npw = _npw_perk[k];
  int memsize = sizeof(fftw_complex)*_nr0*_nr1*_nr2;
  fftw_complex *in = (fftw_complex*)fftw_malloc(memsize);
  fftw_complex *out = (fftw_complex*)fftw_malloc(memsize);
  fftw_plan p;

  for (int nb = 0; nb < _nbands; nb++) 
  {
    struct timeval start, end;   double dt;   gettimeofday(&start, NULL);
    for (int l = 0; l < _nr0*_nr1*_nr2; l++)
    {
      in[l][0] = 0.;
      in[l][1] = 0.;
    }

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
	      m0 += _nr0;
	    int m1 = _mill(1, ik);
	    if (m1 < 0)
	      m1 += _nr1;
	    int m2 = _mill(2, ik);
	    if (m2 < 0)
	      m2 += _nr2;

	    // Miller indices run from -n/2 to n/2
	    // m0, m1, m2 run from 0 to _nr0, _nr1, _nr2
	    // with negative values refolded so they lie
	    // in the "far side of the cell" in G space

	    in[m0*_nr0*_nr1 + m1*_nr1 + m2][0] = _eigvecs[i*_nbands + nb];
	    in[m0*_nr0*_nr1 + m1*_nr1 + m2][1] = 0;
	  }
	}
      }
    }

    // Timing:
    gettimeofday(&end, NULL);
    dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    printf("\t\tTime (sec) to rearrange indices: %g\n", dt);

    gettimeofday(&start, NULL);

    p = fftw_plan_dft_3d(_nr0, _nr1, _nr2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    gettimeofday(&end, NULL);
    dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    printf("\t\tTime (sec) CPU FFT : %g\n", dt);

    gettimeofday(&start, NULL);

    for (int i = 0; i < _nr0; i++) {
      for (int j = 0; j < _nr1; j++) {
	for (int kk = 0; kk < _nr2; kk++) {
	  // Factor of 2 for spin degeneracy. 1/_vol comes from def of plane waves
	  _rhoout(i,j,kk) += (double)2*_wk[k]*std::abs(out[i + j*_nr0 + kk*_nr0*_nr1][0]*
						       out[i + j*_nr0 + kk*_nr0*_nr1][0] + 
						       out[i + j*_nr0 + kk*_nr0*_nr1][1]*
						       out[i + j*_nr0 + kk*_nr0*_nr1][1] )/_vol;
	}
      }
    }

    gettimeofday(&end, NULL);
    dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    // printf("---------Time (sec) for last loop: %g\n", dt);
  }
  fftw_destroy_plan(p);
  fftw_free(in);
  fftw_free(out);
}


double cell::_mix_charge(void)
{
  double drho2 = 0;
  for (int i = 0; i < _nr0; i++)
    for (int j = 0; j < _nr1; j++)
      for (int k = 0; k < _nr2; k++)
	drho2 += pow(std::abs(_rhoout(i,j,k) - _rhoin(i,j,k)),2);

  drho2 = sqrt(drho2 * _vol / (_nr0*_nr1*_nr2));

  for (int i = 0; i < _nr0; i++) {
    for (int j = 0; j < _nr1; j++) {
      for (int k = 0; k < _nr2; k++) {
	_rhoin(i,j,k) = _alpha*_rhoin(i,j,k) + (1. - _alpha)*_rhoout(i,j,k);
      }
    }
  }

  return drho2;
}


void cell::_v_of_rho(void)
{

  // Initialization of cuFFT stuff
  int memsize = sizeof(fftw_complex)*_nr0*_nr1*_nr2;
  fftw_complex *out = (fftw_complex*)fftw_malloc(memsize);
  fftw_plan p;

  // Compute Exchange-Correlation Potential in real space
  fftw_complex zero; zero[0] = 0., zero[1] = 0;
  _vr.Initialize(_nr0,_nr1,_nr2);
  for (int i = 0; i < _nr0; i++) {
    for (int j = 0; j < _nr1; j++) {
      for (int k = 0; k < _nr2; k++) {
	double onethird = 1./3.;
	_vr(i,j,k)[0] = -_e2*pow(3.*_rhoin(i,j,k)/M_PI, onethird);
	_vr(i,j,k)[1] = 0.;
      }
    }
  }

  // Take FFT of V(r) -> V(G)
  p = fftw_plan_dft_3d(_nr0, _nr1, _nr2, _vr.a, out, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p);

  // Now need to divide by grid dimensions!
  for (int i = 0; i < _nr0; i++) {
    for (int j = 0; j < _nr1; j++) {
      for (int k = 0; k < _nr2; k++) {
	_vr(i,j,k)[0] = out[i*_nr0*_nr1 + j*_nr1 + k][0]/(_nr0*_nr1*_nr2);
	_vr(i,j,k)[1] = out[i*_nr0*_nr1 + j*_nr1 + k][1]/(_nr0*_nr1*_nr2);
      }
    }
  }

  // Now get _vg from _vr
  _vg.resize(_npw, 0.);
  for (int i = 0; i < _npw; i++)
  {
    int m0 = _mill(0, i);
    if (m0 < 0)
      m0 += _nr0;
    int m1 = _mill(1, i);
    if (m1 < 0)
      m1 += _nr1;
    int m2 = _mill(2, i);
    if (m2 < 0)
      m2 += _nr2;
    
    _vg[i] = (double)_vr(m0,m1,m2)[0];
  }

  // Need a new vector to store \rho(G) and a way to store the v(G) from v(r)
  std::vector<double> rhog(_npw, 0.);
  Array3D<fftw_complex> vg_;
  vg_.Initialize(_nr0, _nr1, _nr2);
  
  for (int i = 0; i < _nr0; i++) {
    for (int j = 0; j < _nr1; j++) {
      for (int k = 0; k < _nr2; k++) {
	vg_(i,j,k)[0] = _rhoin(i,j,k);
	vg_(i,j,k)[1] = 0.;
      }
    }
  }

  // To compute Hartree potential, have to bring \rho(G) to G-space
  p = fftw_plan_dft_3d(_nr0, _nr1, _nr2, vg_.a, out, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p);

  // Now need to divide by grid dimensions!
  for (int i = 0; i < _nr0; i++) {
    for (int j = 0; j < _nr1; j++) {
      for (int k = 0; k < _nr2; k++) {
	vg_(i,j,k)[0] = out[i*_nr0*_nr1 + j*_nr1 + k][0]/(_nr0*_nr1*_nr2);
	vg_(i,j,k)[1] = out[i*_nr0*_nr1 + j*_nr1 + k][1]/(_nr0*_nr1*_nr2);
      }
    }
  }

  // Now get \rho(G) from vg_
  for (int i = 0; i < _npw; i++)
  {
    int m0 = _mill(0, i);
    if (m0 < 0)
      m0 += _nr0;
    int m1 = _mill(1, i);
    if (m1 < 0)
      m1 += _nr1;
    int m2 = _mill(2, i);
    if (m2 < 0)
      m2 += _nr2;
    
    rhog[i] = (double)vg_(m0,m1,m2)[0];
  }

  // printf("Check: rho(G=0) ?= nelec/volume: %5.5f  ?=  %5.5f\n", rhog[0]*_vol, _nelec/_vol);
  
  for (int i = 0; i < _npw; i++)
    if (_G2[i] > _eps)
      _vg[i] += 4*M_PI*_e2*rhog[i] / _G2[i];

  fftw_destroy_plan(p);
  fftw_free(out);
}


void cell::_scf(void)
{
  // Timing:
  struct timeval start, end;   double dt;   gettimeofday(&start, NULL);

  _nbands = 4;
  _nelec = 8;
  _max_iter = 3;
  _alpha = 0.5; // Charge mixing parameter
  _threshold = 1.e-6; // Convergence threshold

  _rhoin.Initialize(_nr0, _nr1, _nr2, _nelec/_vol);

  _vg.resize(_npw, 0.);

  for (int iter = 0; iter < _max_iter; iter++)
  {
    _rhoout.Initialize(_nr0, _nr1, _nr2, 0.);

    for (int k = 0; k < _nk; k++)
    {
      struct timeval start1, end1; double dt1; gettimeofday(&start1, NULL);
      _fillH(k);
      gettimeofday(&end1, NULL);
      dt1 = ((end1.tv_sec  - start1.tv_sec) * 1000000u + end1.tv_usec - start1.tv_usec) / 1.e6;
      printf("\tTime (sec) for _fillH: %g\n", dt1);

      gettimeofday(&start1, NULL);
      double dt = _diagH(k);
      // printf("Iteration %d, Diagonalization time = %g sec      ", iter, dt);
      gettimeofday(&end1, NULL);
      dt1 = ((end1.tv_sec  - start1.tv_sec) * 1000000u + end1.tv_usec - start1.tv_usec) / 1.e6;
      printf("\tTime (sec) for _diagH: %g\n", dt1);

      gettimeofday(&start1, NULL);
      _sumCharge(k);
      gettimeofday(&end1, NULL);
      dt1 = ((end1.tv_sec  - start1.tv_sec) * 1000000u + end1.tv_usec - start1.tv_usec) / 1.e6;
      printf("\tTime (sec) for _sumCharge: %g\n", dt1);
    }

    struct timeval start1, end1; double dt1; gettimeofday(&start1, NULL);

    double charge = 0;
    for (int i = 0; i < _nr0; i++)
      for (int j = 0; j < _nr1; j++)
	for (int k = 0; k < _nr2; k++)
	  charge += std::abs(_rhoout(i,j,k)) * _vol/(_nr0*_nr1*_nr2);
    
    // printf("Charge: %8.8f\n", charge);

    double drho2 = _mix_charge();
    if ( drho2 < _threshold)
    {
      printf("Convergence threshold %g reached\n", _threshold);
      break;
    }
    // else 
    //   printf("Delta rho = %10.3e\n", drho2);
   
    _v_of_rho();
 
    gettimeofday(&end1, NULL);
    dt1 = ((end1.tv_sec  - start1.tv_sec) * 1000000u + end1.tv_usec - start1.tv_usec) / 1.e6;
    printf("\tTime (sec) for all charge mixing stuff: %g\n", dt1);
  }

  // Timing:
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  printf("Total runtime (sec) for _scf: %g\n", dt);
}
