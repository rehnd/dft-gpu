#ifndef CELL_HPP
#define CELL_HPP

#include <stdlib.h>
#include <vector>
#include <string>
#include <map>
#include <sstream>
#include <Eigen/Dense>

#include <cula.h>
#include <cula_lapack.h>
#include <cufft.h>
#include <fftw3.h>

#include "IndG.hpp"
#include "Array3d.hpp"
#include "tools.hpp"

using namespace Eigen;
using std::string;
using std::vector;


class cell
{
private:
  // Parameters to be used:
  double _ecut;                // Energy cutoff
  double _latconst;            // Lattice constant
  double _a;                   // Possibly use this for latconst instead

  double _tau0, _tau1, _tau2;  // Two atoms are at +tau and -tau (units of a)
  Vector3d _a1, _a2, _a3;      // Direct lattice vectors
  Vector3d _b1, _b2, _b3;      // Reciprocal lattice vectors
  double _vol;                 // Volume (\Omega) of direct lattice

  MatrixXd _G;                 // G vectors (plane waves)
  std::vector<double> _G2;     // G^2 values
  MatrixXi _mill;              // Miller Indices of G vectors
  std::vector<double> _SG;     // Geometrical structure factor S(G)

  MatrixXd _igk;               // Index of G vector in list of k+G such that (k+G)^2 < Ecut

  int _npw;                    // Total Number of plane waves
  std::vector<int> _npw_perk;  // Number of plane waves for each k point
  int _npw_max;                // Maximum number of k+G plane waves

  std::vector<double> _H;      // Hamiltonian

  vector<double> _eigvecs;     // Eigenvectors of H
  vector<double> _eigvals;     // Eigenvalues of H

  int _nm0, _nm1, _nm2;        // Maximum values of Miller Indices
  int _nr0, _nr1, _nr2;        // Real-space Grid Dimensions
  int _nk;                     // Number of k-points in Brillouin Zone
  vector<double> _wk;          // Weights of k points
  MatrixXd _k;                 // k-points in the Brillouin Zone
  // std::map<string, int> _indg; // Gives the index of a G vector from its Miller indices
  IndG _indg;

  double _eps;                 // Small quantity for calculation of vsg
  double _e2;                  // Conversion factor between Rydbergs and Hartrees
  double _alpha;               // Mixing parameter for SCF loop
  double _threshold;           // Threshold for charge comparison
  int _nelec;                  // Number of electrons per unit cell
  int _nbands;                 // Number of occupied bands
  int _max_iter;               // Max # of SCF iterations

  Array3D<double> _rhoin;      // Input charge density (real space)
  Array3D<double> _rhoout;     // Output charge density (real space)
  Array3D<fftw_complex> _vr;   // Real-space exchange and coulomb potential
  vector<double> _vg;          // Reciprocal space potential

  // Internal methods
  void   _get_plane_waves();
  void   _get_SG();
  void   _count_nk();
  void   _fillH(int k);
  double _form_factor(double G2);
  double _diagH(int k);
  void   _calcRho(int k);
  void   _sumCharge(int k);
  double _mix_charge(void);
  void   _v_of_rho(void);

public:
  cell(double ecut, double latconst, int nk);
  void _scf();
};

#endif /* CELL_HPP */
