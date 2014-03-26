
#include "utils/cudaUtils.hpp"
#include "memoryManager/CPUMemory.hpp"

template <typename T>
PinnedCPUResource<T>::PinnedCPUResource() : CPUResource<T>()
{
}

template <typename T>
PinnedCPUResource<T>::PinnedCPUResource(T *data, unsigned int size, bool owner) :
CPUResource<T>(data, size, owner)
{
	this->free();
}

template <typename T>
PinnedCPUResource<T>::~PinnedCPUResource() {
}
			
template <typename T>
const std::string PinnedCPUResource<T>::getResourceType() const {
	return std::string("Pinned CPU Memory");
}


template <typename T>
void PinnedCPUResource<T>::free() {
	
	if(this->_isCPUResource && this->_isOwner) {
		CPUMemory::free<T>(this->_data, this->_size, true);
	}

	this->_data = 0;
	this->_size = 0;
	this->_isOwner = false;
	this->_isCPUResource = false;
}
