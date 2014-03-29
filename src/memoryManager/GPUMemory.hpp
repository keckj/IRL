
#ifndef __MAX_RUNTIME_MEMORY 
#define __MAX_RUNTIME_MEMORY 0 //(1024ul*1024*256)
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

		static bool canAllocateBytes(unsigned long nBytes);

		template <typename T>
		static bool canAllocate(unsigned long nData, int deviceId);

		template <typename T>
		static T *malloc(unsigned long nData, int deviceId);

		template <typename T>
		static void free(T *data, unsigned long nData, int deviceId);
		
		static unsigned long getMinAvailableMemoryOnDevices();

		static void display(std::ostream &out);
			
		static void setVerbose(bool verbose);

	private:
		GPUMemory();
		
		static int _nDevice;
		static const unsigned long * _memorySize;
		static unsigned long * _memoryLeft;
		static unsigned long * _memoryRuntime;
		static bool _verbose;
};

#include "GPUMemory.tpp"

#endif /* end of include guard: GPUMEMORY_H */
