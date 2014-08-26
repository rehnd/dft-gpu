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

#include <cula.h>
#include <cula_lapack.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
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
     a_0 which should multiply each component. This, along with
     computation of the structure factor is very contrived for now -
     basically following a code until I can find a better description
     of calculating S(G) */
  _tau0 = 0.125; _tau1 = 0.125; _tau2 = 0.125; 

  // Define lattice vectors
  _a1 << 0.5*latconst, 0.5*latconst, 0;
  _a2 << 0, 0.5*latconst, 0.5*latconst;
  _a3 << 0.5*latconst, 0, 0.5*latconst;

  // Compute reciprocal lattice vectors
  _vol = std::abs(_a1.dot(_a2.cross(_a3))); // _vol < 0 -> use abs()
  _b1 = (2*M_PI/_vol)*_a2.cross(_a3);
  _b2 = (2*M_PI/_vol)*_a3.cross(_a1);
  _b3 = (2*M_PI/_vol)*_a1.cross(_a2);

  // k point used is a mean-value point
  _k(0,0) = 2*M_PI*0.6223/_latconst; 
  _k(1,0) = 2*M_PI*0.2953/_latconst;
  _k(2,0) = 0.0;

  _get_plane_waves();
  // _G = Map<MatrixXd>(_G.data(),3,_npw);  // Remap this to a 3 x _npw matrix

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
  printf("Number of plane waves: %d\n", _npw);

  // Fill Miller Indices into indg:
  _nm0 = _mill.rowwise().maxCoeff()(0);
  _nm1 = _mill.rowwise().maxCoeff()(1);
  _nm2 = _mill.rowwise().maxCoeff()(2);
  // _indg.Initialize(_nm0, _nm1, _nm2);

  for (int i = 0; i < _npw; i++)
  {
    string str = miller( _mill(0,i), _mill(1,i), _mill(2,i) );
    _indg[str] = i;
    // _indg(_mill(0,i), _mill(1,i), _mill(2,i)) = i;
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

  for (int i = 0; i < _npw; i++) {
    // printf ("%12.8f, %12.8f, %12.8f\n",
    // 	    _G(0,i)/std::abs(_b1(0)) , 
    // 	    _G(1,i)/std::abs(_b1(0)) , 
    // 	    _G(2,i)/std::abs(_b1(0)) );
    _SG[i] = 2.0*cos( 2*M_PI*(_G(0,i)*_tau0/std::abs(_b1(0)) + 
			      _G(1,i)*_tau1/std::abs(_b1(0)) + 
			      _G(2,i)*_tau2/std::abs(_b1(0)) )) / _vol ;
    // printf("SG[%d] = %12.6f\n", i, _SG[i]);
  }
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
  _H.resize(_npw_perk[k] * _npw_perk[k], 0.);

  for (int i = 0; i < _npw_perk[k]; i++)
  {
    int ik = _igk(i, k); // May get more caching if switch i, k
    Vector3d kg = _k.col(k) + _G.col(ik);

    for (int j = i; j < _npw_perk[k]; j++)
    {
      int jk = _igk(j, k);

      int n1 = _mill(0, ik) - _mill(0, jk);
      int n2 = _mill(1, ik) - _mill(1, jk);
      int n3 = _mill(2, ik) - _mill(2, jk);

      string str = miller(n1,n2,n3);
      int ng = _indg[str];
      //int ng = _indg(n1, n2, n3);
      
      double vsg = _form_factor( _G2[ng] );

      if (i == j)
	_H[i + j*_npw_perk[k]] = kg.squaredNorm() + vsg * _SG[ng] + _vg[ng];
      else
	_H[i + j*_npw_perk[k]] = vsg * _SG[ng] + _vg[ng];
      // printf("H(%d,%d) = %g\n", i, j, _H[i+j*_npw_perk[k]]);
    }
  }
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
      _eigvecs[i + j*npw] = val;
      _eigvals[j] = w[j];
    }
  }

  printf("Eigenvalues:   ");
  for (int i = 0; i < _eigvals.size(); i++)
    printf("%g  ", _eigvals[i]);
  printf("\n");

  return dt;
}


// void cell::_calcRho(int k)
// {
//   int npw = _npw_perk[k];
 
//   /* Calculate \rho in reciprocal space:
//      \rho(G) = {1\over \Omega} \sum_{G'}\psi^*(G'-G)\psi(G') */
//   for (int i = 0; i < npw; i++)
//   {
//     int ik = _igk(i, k);
//     for (int j = 0; j < npw; j++)
//     {
//       int jk = _igk(j, k);

//       int n1 = _mill(0, ik) - _mill(0, jk);
//       int n2 = _mill(1, ik) - _mill(1, jk);
//       int n3 = _mill(2, ik) - _mill(2, jk);

//       string str = miller(n1,n2,n3);
//       int ng = _indg[str];
//       // int ng = _indg(n1, n2, n3);

//       for (int nb = 0; nb < _nbands; nb++)
//       {
// 	_rhoout[ng] += _wk[k]*_eigvecs[i*_nbands + nb]*_eigvecs[j*_nbands + nb];
// 	// Would need conjugate for complex Hamiltonians
//       }
//     }
//   }
// }


void cell::_sumCharge(int k)
{
  int npw = _npw_perk[k];

  int memsize = sizeof(fftw_complex)*_nr0*_nr1*_nr2;
  fftw_complex *in  = (fftw_complex*)fftw_malloc(memsize);
  fftw_complex *out = (fftw_complex*)fftw_malloc(memsize);
  fftw_plan p;

  for (int nb = 0; nb < _nbands; nb++)
  {
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

	    in[m0 + m1*_nr0 + m2*_nr0*_nr1][0] = _eigvecs[i + npw*nb];
	    in[m0 + m1*_nr0 + m2*_nr0*_nr1][1] = 0;
	  }
	}
      }
    }
    
    p = fftw_plan_dft_3d(_nr0, _nr1, _nr2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    for (int i = 0; i < _nr0; i++)
    {
      for (int j = 0; j < _nr1; j++)
      {
	for (int kk = 0; kk < _nr2; kk++)
	{
	  // Factor of 2 for spin degeneracy. 1/_vol comes from def of plane waves
	  _rhoout(i,j,kk) += 2*_wk[k]*(out[i + j*_nr0 + kk*_nr0*_nr1][0]*
				       out[i + j*_nr0 + kk*_nr0*_nr1][0] +
				       out[i + j*_nr0 + kk*_nr0*_nr1][1]*
				       out[i + j*_nr0 + kk*_nr0*_nr1][1])/_vol;
	}
      }
    }
  }
  fftw_destroy_plan(p);
  fftw_free(in);
  fftw_free(out);
}


void cell::_scf(void)
{
  // Timing:
  struct timeval start, end;   double dt;   gettimeofday(&start, NULL);

  _nbands = 4;
  _nelec = 8;
  _max_iter = 1;
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
      printf("Time (sec) for _fillH: %g\n", dt1);

      gettimeofday(&start1, NULL);
      double dt = _diagH(k);
      // printf("Iteration %d, Diagonalization time = %g sec      ", iter, dt);
      gettimeofday(&end1, NULL);
      dt1 = ((end1.tv_sec  - start1.tv_sec) * 1000000u + end1.tv_usec - start1.tv_usec) / 1.e6;
      printf("Time (sec) for _diagH: %g\n", dt1);

      gettimeofday(&start1, NULL);
      _sumCharge(k);
      gettimeofday(&end1, NULL);
      dt1 = ((end1.tv_sec  - start1.tv_sec) * 1000000u + end1.tv_usec - start1.tv_usec) / 1.e6;
      printf("Time (sec) for _sumCharge: %g\n", dt1);
    }

    struct timeval start1, end1; double dt1; gettimeofday(&start1, NULL);
    // Account for factor 2 (spin degeneracy) in rhoout:
    // for (int i = 0; i < _npw; i++)
    //   _rhoout[i] = 2. * _rhoout[i] / _vol;

    // Charge mixing

    double charge = 0;
    for (int i = 0; i < _nr0; i++)
      for (int j = 0; j < _nr1; j++)
	for (int k = 0; k < _nr2; k++)
	  charge += std::abs(_rhoout(i,j,k)) * _vol/(_nr0*_nr1*_nr2);
    
    std::cout << "Charge: " << charge << std::endl;

    // double drho2 = 0.;
    // for (int i = 0; i < _npw; i++)
    //   drho2 += pow(_rhoout[i] - _rhoin[i], 2);

    // if ( sqrt(drho2) < _threshold)
    // {
    //   printf("Convergence threshold %g reached\n", _threshold);
    //   break;
    // }
    // else 
    //   printf("Delta rho = %g\n", sqrt(drho2));
    
    // for (int i = 0; i < _npw; i++)
    //   _rhoin[i] = _alpha * _rhoin[i] + (1.-_alpha)*_rhoout[i];

    // New charge is now in rhoout. Calculate new potential hartree term in G space
    // for (int ng = 0; ng < _npw; ng++)
    // {
    //   if (_G2[ng] > _eps)
    // 	_vg[ng] = 4*M_PI*_e2*_rhoin[ng]/_G2[ng];
    //   else
    // 	_vg[ng] = 0.;
    // }
    gettimeofday(&end1, NULL);
    dt1 = ((end1.tv_sec  - start1.tv_sec) * 1000000u + end1.tv_usec - start1.tv_usec) / 1.e6;
    std::cout << "Time (sec) for all charge mixing stuff: " << dt1 << std::endl;
  }

  // Timing:
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  std::cout << "Time (sec) for _scf: " << dt << std::endl;

}
