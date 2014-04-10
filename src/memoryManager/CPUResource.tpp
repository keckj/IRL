#include <cassert>
#include <iostream>
#include <typeinfo>

template <typename T>
CPUResource<T>::CPUResource(unsigned long size) :
_data(0), _size(size), _isOwner(false), _isCPUResource(false)
{
}
	
template <typename T>
CPUResource<T>::CPUResource(CPUResource &original) :
_data(original.data()), _size(original.size()), _isOwner(false), _isCPUResource(original.isCPUResource()) {
}

template <typename T>
CPUResource<T>::CPUResource(T *data, unsigned int size, bool owner) :
_data(data), _size(size), _isOwner(owner), _isCPUResource(size != 0) {
	assert((data == 0 && size == 0) || (data != 0 && size != 0));
}

template <typename T>
CPUResource<T>::~CPUResource() {
}

template <typename T>
T* CPUResource<T>::data() const {
	return _data;
}

template <typename T>
unsigned long CPUResource<T>::size() const {
	return _size;
}

template <typename T>
unsigned long CPUResource<T>::bytes() const {
	return _size * sizeof(T);
}

template <typename T>
bool CPUResource<T>::isOwner() const {
	return _isOwner;
}

template <typename T>
bool CPUResource<T>::isCPUResource() const {
	return _isCPUResource;
}


template <typename T>
void CPUResource<T>::setData(T* data, unsigned int size, bool isOwner) {
	assert((data == 0 && size == 0) || (data != 0 && size != 0));
	assert(_isOwner != true);

	_data = data;
	_size = size;
	_isOwner = isOwner;
	_isCPUResource = true;
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const CPUResource<T> &resource) {
	out << "::CPURessource::" << std::endl;
	out << "\t Is CPU Ressource : " << resource.isCPUResource() << std::endl;
	out << "\t Ressource type : " << resource.getResourceType() << std::endl;
	out << "\t Data : " << typeid(T).name() << std::endl;
	out << "\t Size : " << resource.size() << std::endl;
	out << "\t Bytes : " << resource.bytes() << std::endl;

	return out;
}


template <typename T>
void CPUResource<T>::setSize(unsigned int size) {
	this->_size = size;
}
