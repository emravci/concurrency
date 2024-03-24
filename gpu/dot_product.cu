// Reference: [1] Unified Memory for CUDA Beginners by Mark Harris
//            [2] Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming.

#include <iostream>
#include <numeric>
#include <chrono>
#include <memory>
#include "include/Vector.cuh"

const int N = 1024 * 1024 * 256;
const int threadsPerBlock = 1024;
const int blocksPerGrid = 256;

template<class Type>
__global__ void fill(UniformMemory::Vector<Type>& array, Type value)
{
    const std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = index; i < array.size(); i += stride) { array[i] = value; }
}

template<class Type>
__global__ void dot(UniformMemory::Vector<Type>& partials, const UniformMemory::Vector<Type>& lhs, const UniformMemory::Vector<Type>& rhs)
{
    __shared__ Type cache[threadsPerBlock];

    const std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    const std::size_t cacheIndex = threadIdx.x;

    auto partial = static_cast<Type>(0);
    for(std::size_t i = index, N = lhs.size(); i < N; i += stride) { partial += lhs[index] * rhs[index]; }
    cache[cacheIndex] = partial;
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    for(std::size_t half = blockDim.x / 2; half != 0; half /= 2)
    {
        if (cacheIndex < half) { cache[cacheIndex] += cache[cacheIndex + half]; }
        __syncthreads();
    }

    if(cacheIndex == 0) { partials[blockIdx.x] = cache[0]; }
}

int main()
{
    using VectorType = UniformMemory::Vector<double>;
    auto ptrVectorOfOnes = std::make_unique<VectorType>(N);
    // surprisingly it takes around 320ms, sequential std::fill on CPU takes around 550ms
    // multithreaded fill might perform better
    fill<<<blocksPerGrid, threadsPerBlock>>>(*ptrVectorOfOnes, 1.0);

    // 9ms total
    // concurrent - 3ms dot according to nvprof but memory allocation takes around 1 or 2sec
    auto ptrPartials = std::make_unique<VectorType>(blocksPerGrid);
    auto begin = std::chrono::high_resolution_clock::now();
    dot<<<blocksPerGrid, threadsPerBlock>>>(*ptrPartials, *ptrVectorOfOnes, *ptrVectorOfOnes);
    cudaDeviceSynchronize();
    // sequential addition on CPU - 6ms probably
    double value = std::accumulate(ptrPartials->cbegin(), ptrPartials->cend(), 0.0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << std::fixed << value << " in " << duration.count() << "ms\n";

    return 0;
}
