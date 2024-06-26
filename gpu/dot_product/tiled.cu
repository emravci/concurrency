// Reference: [1] Unified Memory for CUDA Beginners by Mark Harris
//            [2] Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming.
//            [3] CUDA C++ Programming Guide

#include <iostream>
#include <numeric>
#include <chrono>
#include <memory>
#include "../include/ManagedMemory/Vector.cuh"
#include "../include/OpenKernel/Vector.cuh"

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
__device__ typename VectorType::ValueType gridStrideDot(const VectorType& lhs, const VectorType& rhs)
{
    using ValueType = typename VectorType::ValueType;
    using SizeType = typename VectorType::SizeType;
    auto partial = static_cast<ValueType>(0);
    for(SizeType i = threadIdx.x + blockIdx.x * blockDim.x, N = lhs.size(); i < N; i += blockDim.x * gridDim.x) { partial += lhs[i] * rhs[i]; }
    return partial;
}

template<class ValueType, class SizeType>
struct Reduce
{   // for reduction size must be exact power of 2
    __device__ Reduce(ValueType *array, SizeType size) : array_{array}, size_{size} {}
    __device__ void operator()(SizeType index)
    {
        for(SizeType half = size_ / 2; half != 0; half /= 2)
        {
            if(index < half) { array_[index] += array_[index + half]; }
            __syncthreads();
        }
    }
    ValueType *array_;
    SizeType size_;
};

template<class VectorType>
__global__ void dot(VectorType& partials, const VectorType& lhs, const VectorType& rhs)
{
    using ValueType = typename VectorType::ValueType;
    using SizeType = typename VectorType::SizeType;

    __shared__ ValueType cache[threadsPerBlock];

    cache[threadIdx.x] = gridStrideDot(lhs, rhs);
    __syncthreads();

    Reduce reduceCache{cache, threadsPerBlock};
    reduceCache(threadIdx.x);
    // Reduce{cache, threadsPerBlock}(threadIdx.x);

    if(threadIdx.x == 0) { partials[blockIdx.x] = cache[0]; }
}

int main()
{
    #if 1
    using VectorType = ManagedMemory::Vector<double>;
    auto ptrVectorOfOnes = std::make_unique<VectorType>(N);
    VectorType& vectorOfOnes = *ptrVectorOfOnes;
    // surprisingly it takes around 450ms, sequential std::fill on CPU takes around 550ms
    // multithreaded fill might perform better
    fill<<<blocksPerGrid, threadsPerBlock>>>(vectorOfOnes, 1.0);

    // 8ms total
    // concurrent - 8ms dot according to nvprof
    auto ptrPartials = std::make_unique<VectorType>(blocksPerGrid);
    VectorType& partials = *ptrPartials;
    auto begin = std::chrono::high_resolution_clock::now();
    dot<<<blocksPerGrid, threadsPerBlock>>>(partials, vectorOfOnes, vectorOfOnes);
    cudaDeviceSynchronize();

    #elif 0
    using VectorType = ManagedMemory::Vector<double>;
    VectorType vectorOfOnes(N);
    // surprisingly it takes around 439ms, sequential std::fill on CPU takes around 550ms
    // multithreaded fill might perform better
    fill<<<blocksPerGrid, threadsPerBlock>>>(vectorOfOnes, 1.0);

    // 11ms total
    // concurrent - 10ms dot according to nvprof
    VectorType partials(blocksPerGrid);
    auto begin = std::chrono::high_resolution_clock::now();
    dot<<<blocksPerGrid, threadsPerBlock>>>(partials, vectorOfOnes, vectorOfOnes);
    cudaDeviceSynchronize();

    #else
    using VectorType = OpenKernel::Vector<double>;
    VectorType vectorOfOnes(N);
    // it takes around 3400ms, sequential std::fill on CPU takes around 550ms
    fill<<<blocksPerGrid, threadsPerBlock>>>(vectorOfOnes, 1.0);

    // 3400ms total
    // concurrent - 12ms dot according to nvprof
    VectorType partials(blocksPerGrid);
    auto begin = std::chrono::high_resolution_clock::now();
    dot<<<blocksPerGrid, threadsPerBlock>>>(partials, vectorOfOnes, vectorOfOnes);
    // it takes more than 3sec
    cudaDeviceSynchronize();
    #endif
    
    // sequential addition on CPU - 150us
    double value = std::accumulate(partials.cbegin(), partials.cend(), 0.0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << std::fixed << value << " in " << duration.count() << "ms\n";
    return 0;
}
