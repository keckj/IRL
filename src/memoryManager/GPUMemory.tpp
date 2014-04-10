
#include "utils/cudaUtils.hpp"
#include "utils/utils.hpp"

template <typename T>
T* GPUMemory::malloc(unsigned long nData, int deviceId) {

	if(_verbose) {
		if(nData == 0) {
			log_console.warn("Trying to allocate a 0 size block !");
		}
		else {
			log_console.infoStream() << "\tAllocating " << toStringMemory(nData*sizeof(T)) << " on device " << deviceId;  
		}
	}
	
	if(_memoryLeft[deviceId] < nData * sizeof(T)) {
		log_console.errorStream() << "Trying to allocate " << toStringMemory(nData*sizeof(T))	<< " on device " << deviceId << " which has only " << toStringMemory(_memoryLeft[deviceId]) << " left !";
		exit(1);
	}
	
	CHECK_CUDA_ERRORS(cudaSetDevice(deviceId));

	T *data;
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &data, nData * sizeof(T)));

	GPUMemory::_memoryLeft[deviceId] -= nData*sizeof(T);

	return data;
}

template <typename T>
void GPUMemory::free(T* data, unsigned long nData, int deviceId) {

	if(_verbose) {
		if(nData == 0) {
			log_console.warn("Trying to allocate a 0 size block !");
		}
		else {
			log_console.infoStream() << "\tFreeing " << toStringMemory(nData*sizeof(T)) << " on device " << deviceId;  
		}
	}

	int currentDevice;
	CHECK_CUDA_ERRORS(cudaGetDevice(&currentDevice));

	CHECK_CUDA_ERRORS(cudaSetDevice(deviceId));
	CHECK_CUDA_ERRORS(cudaFree(data));
	CHECK_CUDA_ERRORS(cudaSetDevice(currentDevice));

	GPUMemory::_memoryLeft[deviceId] += nData*sizeof(T);
}

template <typename T>
bool GPUMemory::canAllocate(unsigned long nData, int deviceId) {
	return (GPUMemory::_memoryLeft[deviceId] >= nData * sizeof(T));
}

