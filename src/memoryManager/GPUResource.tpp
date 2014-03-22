
#include <cassert>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"

#include "utils/cudaUtils.hpp"
#include "memoryManager/GPUMemory.hpp"

template <typename T>
GPUResource<T>::GPUResource() :
_data(0), _deviceId(0), _size(0), _isOwner(false), _isGPUResource(false)
{
}
	
template <typename T>
GPUResource<T>::GPUResource(GPUResource<T> &original) :
_data(original.data()), _deviceId(original.deviceId()), _size(original.size()), 
_isOwner(false), _isGPUResource(original.isGPUResource()) {
}

template <typename T>
GPUResource<T>::GPUResource(T *data, int deviceId, unsigned int size, bool owner) :
_data(data), _deviceId(deviceId), _size(size), _isOwner(owner), _isGPUResource(true) {
	assert((data == 0 && size == 0) || (data != 0 && size != 0));
}

template <typename T>
GPUResource<T>::~GPUResource() {
	if(_isGPUResource && _isOwner) {
		GPUMemory::free<T>(_data, _size, _deviceId);
	}
}

template <typename T>
T* GPUResource<T>::data() const {
	return _data;
}

template <typename T>
unsigned int GPUResource<T>::size() const {
	return _size;
}

template <typename T>
unsigned int GPUResource<T>::bytes() const {
	return _size * sizeof(T);
}

template <typename T>
bool GPUResource<T>::isOwner() const {
	return _isOwner;
}

template <typename T>
bool GPUResource<T>::isGPUResource() const {
	return _isGPUResource;
}

template <typename T>
int GPUResource<T>::deviceId() const {
	return _deviceId;
}

template <typename T>
const std::string GPUResource<T>::getResourceType() const {
	const std::string str("Device array");
	return str;
}

template <typename T>
void GPUResource<T>::setData(T* data, int deviceId, unsigned int size, bool isOwner) {
	assert((data == 0 && size == 0) || (data != 0 && size != 0));
	assert(_isOwner != true);

	_data = data;
	_deviceId = deviceId;
	_size = size;
	_isOwner = isOwner;
	_isGPUResource = true;
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const GPUResource<T> &resource) {
	out << "::GPURessource::" << std::endl;
	out << "\t Is GPU Ressource : " << resource.isGPUResource() << std::endl;
	out << "\t Device ID : " << resource.deviceId() << std::endl;
	out << "\t Ressource type : " << resource.getResourceType() << std::endl;
	out << "\t Data : " << typeid(T).name() << std::endl;
	out << "\t Size : " << resource.size() << std::endl;
	out << "\t Bytes : " << resource.bytes() << std::endl;

	return out;
}
