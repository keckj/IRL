
template <typename T>
PagedCPUResource<T>::PagedCPUResource(unsigned long size) : CPUResource<T>(size)
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
	return std::string("Paged CPU Memory");
}

template <typename T>
void PagedCPUResource<T>::free() {
	
	if(this->_isCPUResource && this->_isOwner) {
		delete [] this->_data;
	}

	this->_data = 0;
	this->_size = 0;
	this->_isOwner = false;
	this->_isCPUResource = false;
}

template <typename T>
void PagedCPUResource<T>::allocate() {
	this->_data = CPUMemory::malloc<T>(this->_size, false);
	this->_isCPUResource = true;
	this->_isOwner = true;
}
