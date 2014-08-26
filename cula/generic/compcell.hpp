#include <stdlib.h>
#include <vector>
#include <Eigen/Dense>

#include <cula.h>
#include <cula_lapack.h>

using namespace Eigen;

class CompCell
{
private:
  void _get_plane_waves(std::vector<double>& plane_waves);
  void _set_H_kin();
  void _set_H_pot();
  void _set_H_hartree();
  double _compute_eigs();
  void _compute_eigs_cula();
  void _update_hamiltonian(Vector3d& delta);
  void _convert_H_to_culaDouble(const std::vector<double> &shiftedH, culaDouble* shiftedH_cula);

  double _ecut;
  double _latconst;

  Vector3d _a1, _a2, _a3;
  Vector3d _b1, _b2, _b3;
  double _vol;

  MatrixXd _plane_waves;
  int _num_plane_waves;

  std::vector<double> _H;

  MatrixXd _eigvecs;
  VectorXd _eigvals;

  MatrixXd _V;

public:
  CompCell(double ecut, double latconst, double initdelta);
  // void crank_nicolson_update(int i, int numbands, double timestep, VectorXd& dtlocs, int solver);
  // void compute_subdiag(int numbands, Vector3d& delta);
  void _scfcalc();


};
