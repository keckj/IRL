
#include <cassert>

#include "CPUMemory.hpp"
#include "utils/cudaUtils.hpp"

const unsigned long GPUMemory::memorySize = __CPU_MAX_MEMORY;
unsigned long GPUMemory::memoryLeft = __CPU_MAX_MEMORY;
		
CPUMemory::CPUMemory() {
}

static const unsigned long CPUMemory::memorySize() const {
	return GPUMemory::memorySize;
}

static unsigned long CPUMemory::remainingMemory() const {
	return GPUMemory::memoryLeft;
}

static bool CPUMemory::canAllocate(unsigned long NBytes) const {
	return (GPUMemory::memoryLeft >= NBytes);
}

static void CPUMemory::allocate(unsigned long NBytes) {
}

static void CPUMemory::deallocate(unsigned long NBytes) {
	assert(CPUMemory::memoryLeft + NBytes <= GPUMemory::memorySize);
	GPUMemory::memoryLeft += NBytes;
}

		
static void *malloc(unsigned long NBytes, bool pagedMemory=false) {
	assert(CPUMemory::memoryLeft >= NBytes);
	GPUMemory::memoryLeft -= NBytes;
	
	void *data;
	if(pagedMemory) {
		CHECK_CUDA_ERRORS(cudaMallocHost(data, nBytes));
	}
	else {
		data = malloc(nBytes);		
	}

	return data;
}

static void free(CPUResource &src, bool force=false) {
		if(src.isOwner() || force) {
			if(src.isPaged()) 
				CHECK_CUDA_ERRORS(cudaFreeHost(src.data()));
			else
				free(src.data());
		}
}
