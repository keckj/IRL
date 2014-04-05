#include <cassert>
#include "utils/cudaUtils.hpp"

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
SharedResource<T,CPUResourceType,GPUResourceType>::SharedResource(int deviceId, unsigned long size) :
	CPUResourceType<T>(size), GPUResourceType<T>(deviceId, size) 
{
}

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
SharedResource<T,CPUResourceType,GPUResourceType>::SharedResource(GPUResourceType<T> gpuResource) :
	CPUResourceType<T>(gpuResource.size()), GPUResourceType<T>(gpuResource) 
{
}

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
SharedResource<T,CPUResourceType,GPUResourceType>::SharedResource(CPUResourceType<T> cpuResource, int deviceId) :
	CPUResourceType<T>(cpuResource), GPUResourceType<T>(deviceId, cpuResource.size())
{
}

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
SharedResource<T,CPUResourceType,GPUResourceType>::SharedResource(CPUResourceType<T> cpuResource, GPUResourceType<T> gpuResource) :
	CPUResourceType<T>(cpuResource), GPUResourceType<T>(gpuResource)
{
	assert(cpuResource.size() == gpuResource.size());
}

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
SharedResource<T,CPUResourceType,GPUResourceType>::SharedResource(SharedResource<T, CPUResourceType, GPUResourceType> &original) :
	CPUResourceType<T>(original), GPUResourceType<T>(original)
{
}

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
T* SharedResource<T,CPUResourceType,GPUResourceType>::hostData() const {
	return this->CPUResourceType<T>::data();
}

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
T* SharedResource<T,CPUResourceType,GPUResourceType>::deviceData() const {
	return this->GPUResourceType<T>::data();
}

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
void SharedResource<T,CPUResourceType,GPUResourceType>::allocateOnHost() {
	assert(!this->CPUResourceType<T>::isCPUResource());
	this->CPUResourceType<T>::allocate();
}

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
void SharedResource<T,CPUResourceType,GPUResourceType>::allocateOnDevice() {
	assert(!this->GPUResourceType<T>::isGPUResource());
	this->GPUResourceType<T>::allocate();
}

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
void SharedResource<T,CPUResourceType,GPUResourceType>::allocateAll() {
	this->allocateOnHost();
	this->allocateOnDevice();
}

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
unsigned long SharedResource<T,CPUResourceType,GPUResourceType>::dataSize() const {
	return this->CPUResourceType<T>::size();
}

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
unsigned long SharedResource<T,CPUResourceType,GPUResourceType>::dataBytes() const {
	return this->CPUResourceType<T>::bytes();
}

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
int SharedResource<T,CPUResourceType,GPUResourceType>::deviceId() const {
	return this->GPUResourceType<T>::deviceId();
}
		
template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
void SharedResource<T, CPUResourceType, GPUResourceType>::copyToDevice(cudaStream_t stream) {
	assert(this->CPUResourceType<T>::isCPUResource());
	assert(this->GPUResourceType<T>::isGPUResource());
	cudaSetDevice(this->deviceId());
	CHECK_CUDA_ERRORS(cudaMemcpyAsync(this->deviceData(), this->hostData(), this->dataBytes(), cudaMemcpyHostToDevice, stream));
}

template <typename T,
	template <typename> class CPUResourceType,
	template <typename> class GPUResourceType>
void SharedResource<T, CPUResourceType, GPUResourceType>::copyToHost(cudaStream_t stream) {
	assert(this->CPUResourceType<T>::isCPUResource());
	assert(this->GPUResourceType<T>::isGPUResource());
	cudaSetDevice(this->deviceId());
	CHECK_CUDA_ERRORS(cudaMemcpyAsync(this->hostData(), this->deviceData(), this->dataBytes(), cudaMemcpyDeviceToHost, stream));
}
