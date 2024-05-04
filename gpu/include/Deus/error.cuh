// Reference: [1] github/cuda-samples/common/helper-cuda.h

#pragma once

#include <iostream>

#define checkCudaError(fcn) Deus::__checkCudaError((fcn), #fcn, __FILE__, __LINE__)

namespace Deus { void __checkCudaError(cudaError_t, const char *const, const char *const, const std::size_t); }

inline void Deus::__checkCudaError(cudaError_t error, const char *const func, const char *const file, const std::size_t line) 
{
    if(error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%zu code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(error), cudaGetErrorString(error), func);
        exit(EXIT_FAILURE);
    }
}
