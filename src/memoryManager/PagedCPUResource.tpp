
template <typename T>
PagedCPUResource<T>::PagedCPUResource() : CPUResource<T>()
{
}

template <typename T>
PagedCPUResource<T>::PagedCPUResource(T *data, unsigned int size, bool owner) :
CPUResource<T>(data, size, owner)
{
}

template <typename T>
PagedCPUResource<T>::~PagedCPUResource() {
	if(this->_isCPUResource && this->_isOwner) {
		CPUMemory::free<T>(this->_data, this->_size, false);
	}
}
			
template <typename T>
const std::string PagedCPUResource<T>::getResourceType() const {
	return string("Paged CPU Memory");
}
