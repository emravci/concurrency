// Reference: [1] Unified Memory in CUDA 6 by Mark Harris. NVIDIA Developer Blog

#pragma once

namespace UnifiedMemory { class Managed; }

class UnifiedMemory::Managed
{
    public: 
    void *operator new(std::size_t len)
    {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    void operator delete(void *ptr)
    {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};
