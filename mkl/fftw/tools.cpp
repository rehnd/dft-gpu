#include <stdio.h>
#include <string>
#include <sstream>


void print_matrix( char* desc, int m, int n, double* a)
{
  int i, j;
  printf("\n%s: \n", desc);
  for( i = 0; i < m; i++ )
  {
    for( j = 0; j < n; j++ ) printf( "%10.6f", a[i*n + j] );
    printf( "\n" );
  }
}


void print_matrix_transpose( char* desc, int m, int n, double* a)
{
  int i, j;
  printf("\n%s: \n", desc);
  for( i = 0; i < m; i++ )
  {
    for( j = 0; j < n; j++ ) printf( "%10.6f", a[i + j*n] );
    printf( "\n" );
  }
}


std::string miller(int i0, int i1, int i2)
{
  /* For lack of better design, store Miller indices in a map
     with key as a string representing the three miller indices */ 
  std::stringstream ss;
  ss << i0 << " " << i1 << " " << i2;
  return ss.str();
}

