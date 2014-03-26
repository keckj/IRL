

template <typename T>
T* GPUMemory::malloc(unsigned long nData, int deviceId) {
	
	assert(GPUMemory::_memoryLeft[deviceId] >= nData * sizeof(T));

	T *data;
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &data, nData * sizeof(T)));
	
	GPUMemory::_memoryLeft[deviceId] -= nData*sizeof(T);

	return data;
}

template <typename T>
void GPUMemory::free(T* data, unsigned long nData, int deviceId) {
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

