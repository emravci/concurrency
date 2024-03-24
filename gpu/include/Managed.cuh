// Reference: [1] Unified Memory in CUDA 6 by Mark Harris. NVIDIA Developer Blog

namespace UniformMemory { class Managed; }

class UniformMemory::Managed
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
