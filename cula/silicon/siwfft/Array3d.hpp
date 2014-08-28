#ifndef ARRAY3D_HPP
#define ARRAY3D_HPP

#include <stdio.h>

template <typename T>
class Array3D
{
  /* This class provides an interface to a 1D static array so that
   * array accesses can be handled like 3D array accesses. This differs 
   * from IndG in that the x,y,z components are 0-based indexing. */
private:
  int _x, _y, _z; // Array runs from [-_x, _x], [-_y, _y], and [-_z, _z]
  int _size;

public:
  T *a;
  Array3D() { }
  Array3D(int x, int y, int z, T value)
  {
    _x = x; _y = y; _z = z;
    _size = _x*_y*_z;
    a = (T*)malloc(_size*sizeof(T));

    for (int i = 0; i < _size; i++)
      a[i] = value;
  }
  void Initialize(int x, int y, int z, T value)
  {
    _x = x; _y = y; _z = z;
    _size = _x*_y*_z;
    a = (T*)malloc(_size*sizeof(T));
    for (int i = 0; i < _size; i++)
      a[i] = value;
  }
  T& operator() (int i, int j, int k)
  {
    int index = i*_x*_y + j*_y + k;
    if (index < _size && index >= 0)
      return a[index];
    else
    {
      printf("Error: Out of bounds access for index (%d, %d, %d)\n", i, j, k);
      exit(1);
    }
  }
  const T& operator() (int i, int j, int k) const
  {
    int index = i*_x*_y + j*_y + k;
    if (index < _size && index >= 0)
      return a[index];
    else
    {
      printf("Error: Out of bounds access for index (%d, %d, %d)\n", i, j, k);
      exit(1);
    }
  }
  ~Array3D()
  {
    free(a);
  }
};

#endif /* ARRAY_3D */
