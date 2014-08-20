#include <vector>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/LU>
#include "compcell.hpp"
#include <sys/time.h>

using namespace Eigen;

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
  /** Start Timer ***/
  gettimeofday(&start, NULL);
  _get_plane_waves(plane_waves);
  /*** End Timer ***/
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  std::cout << "Time (sec) for _get_plane_waves: " << dt << std::endl;
  _plane_waves = Map<MatrixXd>(plane_waves.data(),3,_num_plane_waves);

  // initialize Hamiltonian
  /** Start Timer ***/
  gettimeofday(&start, NULL);
  _H.resize(_num_plane_waves,_num_plane_waves);
  _set_H_kin();
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  std::cout << "Time (sec) for initialize H_kin Hamiltonian: " << dt << std::endl;
  
  gettimeofday(&start, NULL);
  Vector3d delta(0, 0, initdist);
  _update_hamiltonian(delta);
  /*** End Timer ***/
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  std::cout << "Time (sec) for initialize H_pot Hamiltonian: " << dt << std::endl;
  
  // compute initial eigenstates
  /** Start Timer ***/
  gettimeofday(&start, NULL);
  _compute_eigs();
  /*** End Timer ***/
  gettimeofday(&end, NULL);
  dt = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  std::cout << "Time (sec) to compute eigenvalues: " << dt << std::endl;
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
  for (int i=0; i<_num_plane_waves; i++) {
    Vector3d k_G = _plane_waves.col(i);
    _H(i,i) = k_G.squaredNorm()/2;
  }
}


void CompCell::_set_H_pot(Vector3d& delta)
{
  for (int i=0; i<_num_plane_waves; i++) {
    for (int j=i+1; j<_num_plane_waves-1; j++) {
      Vector3d delta_G = _plane_waves.col(i) - _plane_waves.col(j);
      double H_orig_pot = -4*M_PI/_vol/delta_G.squaredNorm();
      double H_new_pot = 2*cos(delta_G.dot(delta))*H_orig_pot;
      
      _H(i,j) = H_orig_pot;
      _H(j,i) = H_orig_pot;
    }
  }
}


void CompCell::_compute_eigs()
{
  // shift for faster computation
  MatrixXd shiftedH = _H - MatrixXd::Identity(_num_plane_waves,_num_plane_waves)*10000;
  SelfAdjointEigenSolver<MatrixXd> eigensolver(_num_plane_waves);
  eigensolver.compute(shiftedH);
  
  _eigvecs = eigensolver.eigenvectors();
  _eigvals = eigensolver.eigenvalues() + 10000*VectorXd::Ones(_num_plane_waves,1);
  
  std::cout << "Lowest eigval: " << _eigvals(0) << std::endl;
}


void CompCell::_update_hamiltonian(Vector3d& delta)
{
  _set_H_pot(delta);
}
