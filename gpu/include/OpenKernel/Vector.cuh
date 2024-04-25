// Reference: [1] CUDA C++ Programming Guide

namespace OpenKernel { template<class Type> class Vector; }

template<class Type>
class OpenKernel::Vector
{
    public:
    Vector(std::size_t size) : data_{new Type[size]}, size_{size} {}
    Vector(const Vector& src) : data_{new Type[src.size_]}, size_{src.size_}
    {
        for(std::size_t i = 0; i < size_; ++i) { data_[i] = src.data_[i]; }
    }
    Vector& operator=(const Vector&) = delete;
    Vector(Vector&&) = delete;
    Vector& operator=(Vector&&) = delete;
    ~Vector() { delete[] data_; }
    __host__ __device__ std::size_t size() const { return size_; }
    __host__ __device__ Type& operator[](std::size_t i) { return data_[i]; }
    __host__ __device__ const Type& operator[](std::size_t i) const { return data_[i]; }
    private:
    Type *data_ = nullptr;
    std::size_t size_ = 0;
};
