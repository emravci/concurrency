// Reference: [1] github/cuda-samples/common/helper-cuda.h

#pragma once

#include <iostream>

#define checkCudaError(fcn) Deus::__checkCudaError((fcn), #fcn, __FILE__, __LINE__)

#define checkLastCudaError(mgs) Deus::__checkLastCudaError(mgs, __FILE__, __LINE__)

namespace Deus 
{ 
    void __checkCudaError(cudaError_t, const char *const, const char *const, const std::size_t);
    void __checkLastCudaError(const char *const, const char *const, const std::size_t);
}

inline void Deus::__checkCudaError(cudaError_t error, const char *const func, const char *const file, const std::size_t line) 
{
    if(error != cudaSuccess)
    {
        fprintf(stderr, "CUDA\e[31m\e[1m error\e[0m at\e[1m %s:%zu\e[0m code=%d(%s) during call \"%s\" \n",
            file, line, static_cast<unsigned int>(error), cudaGetErrorString(error), func);
        exit(EXIT_FAILURE);
    }
}

inline void Deus::__checkLastCudaError(const char *const message, const char *const file, const std::size_t line)
{
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        fprintf(stderr, "CUDA\e[31m\e[1m error\e[0m at\e[1m %s:%zu\e[0m code=%d(%s) with\e[1m \"%s\"\e[0m\n",
            file, line, static_cast<unsigned int>(error), cudaGetErrorString(error), message);
        exit(EXIT_FAILURE);
    }
}
