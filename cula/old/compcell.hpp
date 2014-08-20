#include <stdlib.h>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class CompCell
{  
public:
  CompCell(double ecut, double latconst, double initdelta);
  // void crank_nicolson_update(int i, int numbands, double timestep, VectorXd& dtlocs, int solver);
  // void compute_subdiag(int numbands, Vector3d& delta);
  
private:
  void _get_plane_waves(std::vector<double>& plane_waves);
  void _set_H_kin();
  void _set_H_pot(Vector3d& delta);
  void _compute_eigs();
  void _update_hamiltonian(Vector3d& delta);
 
  double _ecut;
  double _latconst;
  
  Vector3d _a1;
  Vector3d _a2;
  Vector3d _a3;
  Vector3d _b1;
  Vector3d _b2;
  Vector3d _b3;
  double _vol;
  
  MatrixXd _plane_waves;
  int _num_plane_waves;
  
  std::vector<double> _H;
  
  double* _eigvecs;
  double* _eigvals;
  
  MatrixXcd _V; // linear combos of eigenvectors during time evolution
};
