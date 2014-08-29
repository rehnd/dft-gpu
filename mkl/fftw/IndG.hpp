#ifndef INDG_HPP
#define INDG_HPP

#include <stdio.h>

class IndG
{
  /* This class accesses, for each set of Miller Indices (i, j, k),
     the corresponding value of m in $\vec G_m$, so that one can find
     a particular G vector from the Miller Indices. */
private:
  int _x, _y, _z; // Array runs from [-_x, _x], [-_y, _y], and [-_z, _z]
  int _xl, _yl, _zl;
  int _size;
  int *a;

public:
  IndG() { }
  IndG(int x, int y, int z)
  {
    _x = x; _y = y; _z = z;
    _xl = 2*_x + 1;  _yl = 2*_y + 1;  _zl = 2*_z + 1;
    _size = _xl*_yl*_zl;
    printf("size = %d\n", _size);
    a = (int*)malloc(_size*sizeof(int));
  }
  void Initialize(int x, int y, int z)
  {
    _x = x; _y = y; _z = z;
    _xl = 2*_x + 1;  _yl = 2*_y + 1;  _zl = 2*_z + 1;
    _size = _xl*_yl*_zl;
    a = (int*)malloc(_size*sizeof(int));
  }
  int& operator() (int i, int j, int k)
  {
    int index = (i + _x)*_xl*_yl + (j + _y)*_yl + (k + _z);
    if (index < _size && index >= 0)
      return a[index];
    else
    {
      printf("Error: Out of bounds access for index (%d, %d, %d)\n", i, j, k);
      exit(1);
    }
  }
  const int& operator() (int i, int j, int k) const
  {
    int index = (i + _x)*_xl*_yl + (j + _y)*_yl + (k + _z);
    if (index < _size && index >= 0)
      return a[index];
    else
    {
      printf("Error: Out of bounds access for index (%d, %d, %d)\n", i, j, k);
      exit(1);
    }
  }
  ~IndG()
  {
    free(a);
  }
};

#endif /* INDG_HPP */
