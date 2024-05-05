// Reference: [1] Unified Memory in CUDA 6 by Mark Harris. NVIDIA Developer Blog

#pragma once

#include "../Deus/error.cuh"

namespace ManagedMemory { class Managed; }

class ManagedMemory::Managed
{
    public: 
    void *operator new(std::size_t len)
    {
        void *ptr;
        checkCudaError(cudaMallocManaged(&ptr, len));
        checkCudaError(cudaDeviceSynchronize());
        return ptr;
    }

    void operator delete(void *ptr)
    {
        checkCudaError(cudaDeviceSynchronize());
        checkCudaError(cudaFree(ptr));
    }
};
