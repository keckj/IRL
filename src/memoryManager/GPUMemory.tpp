
template <typename T>
T* CPUMemory::malloc(unsigned int nData, bool pinnedMemory) {
	
	assert(CPUMemory::_memoryLeft >= nData * sizeof(T));

	T *data;
	if(pinnedMemory) {
		CHECK_CUDA_ERRORS(cudaMallocHost(data, nData * sizeof(T)));
	}
	else {
		data = new T[nData];		
	}
	
	CPUMemory::_memoryLeft -= nData*sizeof(T);

	return data;
}

template <typename T>
void CPUMemory::free(T* data, unsigned int nData, bool pinnedMemory) {
	if(pinnedMemory) { 
		CHECK_CUDA_ERRORS(cudaFreeHost(data));
	}
	else {
		delete [] data;
	}
	
	CPUMemory::_memoryLeft += nData*sizeof(T);
}

template <typename T>
bool canAllocate(unsigned int nData) {
	return (CPUMemory::_memoryLeft >= nData * sizeof(T));
}

