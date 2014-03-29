
#include "cuda.h"
#include "cuda_runtime.h"
#include "utils/cudaUtils.hpp"
#include "utils/log.hpp"
#include "utils/utils.hpp"

template <typename T>
T* CPUMemory::malloc(unsigned long nData, bool pinnedMemory) {

	assert(CPUMemory::_memoryLeft >= nData * sizeof(T));

	if(_verbose) {
		if(nData == 0) {
			log_console.warn("Trying to allocate a 0 size block !");
		}
		else {
			log_console.infoStream() << "\tAllocating " << toStringMemory(nData*sizeof(T)) << " on host !";  
		}
	}

	T *data;
	if(pinnedMemory) {
		CHECK_CUDA_ERRORS(cudaMallocHost((void **) &data, nData * sizeof(T)));
	}
	else {
		data = new T[nData];		
	}

	CPUMemory::_memoryLeft -= nData*sizeof(T);

	return data;
}

template <typename T>
void CPUMemory::free(T* data, unsigned long nData, bool pinnedMemory) {

	if(_verbose) {
		if(nData == 0) {
			log_console.warn("Trying to free a 0 size block !");
		}
		else {
			log_console.infoStream() << "\tFreeing " << toStringMemory(nData*sizeof(T)) << " on host !";  
		}
	}

	if(pinnedMemory) { 
		CHECK_CUDA_ERRORS(cudaFreeHost(data));
	}
	else {
		delete [] data;
	}

	CPUMemory::_memoryLeft += nData*sizeof(T);
}

template <typename T>
bool CPUMemory::canAllocate(unsigned long nData) {
	return (CPUMemory::_memoryLeft >= nData * sizeof(T));
}


