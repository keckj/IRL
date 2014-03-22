
#include <cassert>

#include "GPUMemory.hpp"
#include "utils/cudaUtils.hpp"
#include "utils/utils.hpp"

int GPUMemory::_nDevice = 0;
const unsigned long * GPUMemory::_memorySize = 0;
unsigned long * GPUMemory::_memoryLeft = 0;
unsigned long * GPUMemory::_memoryRuntime = 0;
		
GPUMemory::GPUMemory() {
}

void GPUMemory::init() {
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	GPUMemory::_nDevice = nDevices;
	GPUMemory::_memorySize = new unsigned long[nDevices];
	GPUMemory::_memoryLeft = new unsigned long[nDevices];
	GPUMemory::_memoryRuntime = new unsigned long[nDevices];

	cudaDeviceProp prop;
	for (int i = 0; i < nDevices; i++) {
		cudaGetDeviceProperties(&prop, i);
		const_cast<unsigned long *>(GPUMemory::_memorySize)[i] = prop.totalGlobalMem;
		GPUMemory::_memoryLeft[i] = prop.totalGlobalMem - __MIN_RUNTIME_MEMORY;
		GPUMemory::_memoryRuntime[i] = __MIN_RUNTIME_MEMORY;
	}

}

unsigned long GPUMemory::memorySize(int deviceId) {
	assert(_nDevice > deviceId);
	return GPUMemory::_memorySize[deviceId];
}

unsigned long GPUMemory::memoryLeft(int deviceId) {
	assert(_nDevice > deviceId);
	return GPUMemory::_memoryLeft[deviceId];
}

		
void GPUMemory::display(std::ostream &out) {
	out << ":: GPU VRAM Status ::" << std::endl; 
	for (int i = 0; i < _nDevice; i++) {
		out << "\t GPU " << i
			<< "\t Total : " << toStringMemory(GPUMemory::_memorySize[i]) 
			<< "\t Reserved : " << toStringMemory(GPUMemory::_memoryRuntime[i])
			<< "\t Used : " << toStringMemory(GPUMemory::_memorySize[i] - GPUMemory::_memoryLeft[i]) 
			<< "\t " << 100*(float)(GPUMemory::_memorySize[i] - GPUMemory::_memoryLeft[i])/GPUMemory::_memorySize[i] << "%"
			<< std::endl; 
	}

}
