// Reference: [1] Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming.

#include <iostream>

namespace CUDA
{
	template<class Type> class FreeFunction;

	template<class ReturnType, class ...Args>
	class FreeFunction<ReturnType(Args...)>
	{
		public:
		FreeFunction(void(*fcn)(ReturnType*, Args...)) : dataOnDevice_{nullptr}, fcn_{fcn}
		{
			allocateMemoryOnDevice();
		}
		const ReturnType& operator()(Args&& ...args)
		{
			callFunctionOnDevice(std::forward<Args>(args)...);
			retrieveResultFromDevice();
			return dataOnHost_;
		}
		~FreeFunction()
		{
			freeMemoryOnDevice();
		}
		private:
		void allocateMemoryOnDevice() { cudaMalloc((void**)&dataOnDevice_, sizeof(ReturnType)); }
		void callFunctionOnDevice(Args&& ...args) { (*fcn_)<<<1, 1>>>(dataOnDevice_, std::forward<Args>(args)...); }
		void retrieveResultFromDevice() { cudaMemcpy(&dataOnHost_, dataOnDevice_, sizeof(ReturnType), cudaMemcpyDeviceToHost); }
		void freeMemoryOnDevice() { cudaFree(dataOnDevice_); }
		private:
		ReturnType dataOnHost_;
		ReturnType* dataOnDevice_;
		void(*fcn_)(ReturnType*, Args...);
	};
}

__global__ void add(int* result, int lhs, int rhs)
{
	*result = lhs + rhs;
}

__global__ void add(double* result, double a, double b, double c, double d)
{
	*result = a + b + c + d;
}

int main()
{
	CUDA::FreeFunction<int(int, int)> addIntegers(add);	
	std::cout << addIntegers(2, 7) << "\n";
	std::cout << addIntegers(3, 10) << "\n";

	CUDA::FreeFunction<double(double, double, double, double)> addDoubles(add);
	std::cout << addDoubles(2.0, 7.0, 3.0, 10.5) << "\n";

	return 0;
}
