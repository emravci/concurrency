#include <iostream>
#include <memory>

#include "../include/Matrix.cuh"
#include "../include/Vector.cuh"

template<class Type>
__global__ void fill(UniformMemory::Vector<Type>& array, Type value)
{
    for(std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < array.size(); i += blockDim.x * gridDim.x) { array[i] = value; }
}

template<class Type>
__global__ void fill(UniformMemory::Matrix<Type>& matrix, Type value)
{
    for(std::size_t i = threadIdx.y + blockIdx.y * blockDim.y; i < matrix.row(); i += blockDim.y * gridDim.y)
    {
        for(std::size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < matrix.column(); j += blockDim.x * gridDim.x)
        {
            matrix(i, j) = value;
        }
    }
}

template<class Type>
__global__ void multiply(UniformMemory::Vector<Type>& ans, const UniformMemory::Matrix<Type>& matrix, const UniformMemory::Vector<Type>& vector)
{
    for(std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < matrix.row(); i += blockDim.x * gridDim.x)
    {
        auto value = static_cast<Type>(0);
        for(std::size_t j = 0; j < matrix.column() && j < vector.size(); ++j)
        {
            value += matrix(i, j) * vector[j];
        }
        ans[i] = value;
    }
}

template<class Type>
bool checkResult(const UniformMemory::Vector<Type>& vector, Type value)
{
    for(std::size_t i = 0; i < vector.size(); ++i) { if(vector[i] != value) { return false; } }
    return true;
}

int main()
{
    using MatrixType = UniformMemory::Matrix<double>;
    using VectorType = UniformMemory::Vector<double>;
    constexpr std::size_t row = 1024 * 2;
    constexpr std::size_t col = 1024 * 4;
    auto pMatrix = std::make_unique<MatrixType>(row, col);
    {   // fill with ones
        dim3 threadsPerBlock(32, 32);
        dim3 blocksPerGrid(32, 32);
        fill<<<blocksPerGrid, threadsPerBlock>>>(*pMatrix, 1.0);
    }
    auto pVector = std::make_unique<VectorType>(col);
    {   // fill with ones
        dim3 threadsPerBlock(1024);
        dim3 blocksPerGrid(1);
        fill<<<blocksPerGrid, threadsPerBlock>>>(*pVector, 1.0);
    }
    auto pAnswer = std::make_unique<VectorType>(row);
    {   // matrix vector multiplication takes 73ms according to nvprof
        dim3 threadsPerBlock(1024);
        dim3 blocksPerGrid(1);
        multiply<<<blocksPerGrid ,threadsPerBlock>>>(*pAnswer, *pMatrix, *pVector);
        cudaDeviceSynchronize();
    }
    std::cout << std::boolalpha << checkResult(*pAnswer, static_cast<double>(col)) << "\n";
	return 0;
}
