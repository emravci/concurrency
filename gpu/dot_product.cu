// Reference: [1] Unified Memory for CUDA Beginners by Mark Harris
//            [2] Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming.

#include <iostream>
#include <numeric>
#include <chrono>
#include <memory>
#include "include/UnifiedMemory/Vector.cuh"
#include "include/OpenKernel/Vector.cuh"

const int N = 1024 * 1024 * 256;
const int threadsPerBlock = 1024;
const int blocksPerGrid = 256;

template<class VectorType>
__global__ void fill(VectorType& array, typename VectorType::ValueType value)
{
    using SizeType = typename VectorType::SizeType;
    for (SizeType i = threadIdx.x + blockIdx.x * blockDim.x; i < array.size(); i += blockDim.x * gridDim.x) { array[i] = value; }
}

template<class VectorType>
__global__ void dot(VectorType& partials, const VectorType& lhs, const VectorType& rhs)
{
    using ValueType = typename VectorType::ValueType;
    using SizeType = typename VectorType::SizeType;

    __shared__ ValueType cache[threadsPerBlock];

    const SizeType cacheIndex = threadIdx.x;

    auto partial = static_cast<ValueType>(0);
    for(SizeType i = threadIdx.x + blockIdx.x * blockDim.x, N = lhs.size(); i < N; i += blockDim.x * gridDim.x) { partial += lhs[i] * rhs[i]; }
    cache[cacheIndex] = partial;
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    for(SizeType half = blockDim.x / 2; half != 0; half /= 2)
    {
        if (cacheIndex < half) { cache[cacheIndex] += cache[cacheIndex + half]; }
        __syncthreads();
    }

    if(cacheIndex == 0) { partials[blockIdx.x] = cache[0]; }
}

int main()
{
    using VectorType = UnifiedMemory::Vector<double>;
    auto ptrVectorOfOnes = std::make_unique<VectorType>(N);
    // surprisingly it takes around 320ms, sequential std::fill on CPU takes around 550ms
    // multithreaded fill might perform better
    fill<<<blocksPerGrid, threadsPerBlock>>>(*ptrVectorOfOnes, 1.0);

    // 8ms total
    // concurrent - 8ms dot according to nvprof
    auto ptrPartials = std::make_unique<VectorType>(blocksPerGrid);
    auto begin = std::chrono::high_resolution_clock::now();
    dot<<<blocksPerGrid, threadsPerBlock>>>(*ptrPartials, *ptrVectorOfOnes, *ptrVectorOfOnes);
    cudaDeviceSynchronize();
    
    // sequential addition on CPU - 150us
    double value = std::accumulate(ptrPartials->cbegin(), ptrPartials->cend(), 0.0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << std::fixed << value << " in " << duration.count() << "ms\n";

    return 0;
}
