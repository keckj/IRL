

template <typename T>
T* GPUMemory::malloc(unsigned int nData, int deviceId) {
	
	assert(GPUMemory::_memoryLeft[deviceId] >= nData * sizeof(T));

	T *data;
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &data, nData * sizeof(T)));
	
	GPUMemory::_memoryLeft[deviceId] -= nData*sizeof(T);

	return data;
}

template <typename T>
void GPUMemory::free(T* data, unsigned int nData, int deviceId) {
	CHECK_CUDA_ERRORS(cudaFree(data));
	GPUMemory::_memoryLeft[deviceId] += nData*sizeof(T);
}

template <typename T>
bool canAllocate(unsigned int nData, int deviceId) {
	return (GPUMemory::_memoryLeft[deviceId] >= nData * sizeof(T));
}

