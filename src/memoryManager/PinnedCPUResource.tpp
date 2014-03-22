
#include "utils/cudaUtils.hpp"

template <typename T>
PinnedCPUResource<T>::PinnedCPUResource() : CPUResource<T>()
{
}

template <typename T>
PinnedCPUResource<T>::PinnedCPUResource(T *data, unsigned int size, bool owner) :
CPUResource<T>(data, size, owner)
{
}

template <typename T>
PinnedCPUResource<T>::~PinnedCPUResource() {
	if(this->_isOwner) {
		CHECK_CUDA_ERRORS(cudaFreeHost(this->_data));
	}
}
			
template <typename T>
const std::string PinnedCPUResource<T>::getResourceType() const {
	return string("Pinned CPU Memory");
}
