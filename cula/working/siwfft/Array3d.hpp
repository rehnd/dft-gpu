#ifndef ARRAY3D_HPP
#define ARRAY3D_HPP

#include <stdio.h>


class Array3d
{
  /* This class accesses, for each set of Miller Indices (i, j, k),
     the corresponding value of m in $\vec G_m$, so that one can find
     a particular G vector from the Miller Indices. */
private:
  int _x, _y, _z; // Array runs from [-_x, _x], [-_y, _y], and [-_z, _z]
  int _size;
  double *a;

public:
  Array3d(int x, int y, int z, double value)
  {
    _x = x; _y = y; _z = z;
    _size = _x*_y*_z;
    a = (double*)malloc(_size*sizeof(double));

    for (int i = 0; i < _size; i++)
      a[i] = value;
  }
  double& operator() (int i, int j, int k)
  {
    int index = i + j*_xl + k*_xl*_yl;
    if (index < _size && index >= 0)
      return a[index];
    else
    {
      printf("Error: Out of bounds access for index (%d, %d, %d)\n", i, j, k);
      exit(1);
    }
  }
  const double& operator() (int i, int j, int k) const
  {
    int index = i + j*_xl + k*_xl*_yl;
    if (index < _size && index >= 0)
      return a[index];
    else
    {
      printf("Error: Out of bounds access for index (%d, %d, %d)\n", i, j, k);
      exit(1);
    }
  }
  ~Array3d()
  {
    free(a);
  }
};

#endif /* ARRAY_3D */
