
#ifndef __MIN_RUNTIME_MEMORY 
#define __MIN_RUNTIME_MEMORY 200000000
#endif

#include <ostream>
#include <cassert>
#include "cuda.h"
#include "cuda_runtime.h"

#include "utils/cudaUtils.hpp"

#ifndef GPUMEMORY_H
#define GPUMEMORY_H

class GPUMemory {
	public:

		static void init();

		static unsigned long memorySize(int deviceId); //bytes
		static unsigned long memoryLeft(int deviceId); //bytes

		static bool canAllocateBytes(unsigned int nBytes);

		template <typename T>
		static bool canAllocate(unsigned int nData, int deviceId);

		template <typename T>
		static T *malloc(unsigned int nData, int deviceId);

		template <typename T>
		static void free(T *data, unsigned int nData, int deviceId);

		static void display(std::ostream &out);
			

	private:
		GPUMemory();
		
		static int _nDevice;
		static const unsigned long * _memorySize;
		static unsigned long * _memoryLeft;
		static unsigned long * _memoryRuntime;
};

#include "GPUMemory.tpp"

#endif /* end of include guard: GPUMEMORY_H */
