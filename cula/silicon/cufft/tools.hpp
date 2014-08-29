#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <cula.h>
#include <cula_lapack.h>
#include <cuda_runtime_api.h>
#include <cuda.h>


void print_matrix( char* desc, int m, int n, double* a);
void print_matrix_transpose( char* desc, int m, int n, double* a);
void checkStatus(culaStatus status);
std::string miller(int i0, int i1, int i2);

#endif /* TOOLS_HPP */
