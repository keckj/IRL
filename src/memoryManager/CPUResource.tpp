#include <cassert>
#include <iostream>

template <typename T>
CPUResource<T>::CPUResource() :
_data(0), _size(0), _isOwner(false), _isCPUResource(false)
{
}

template <typename T>
CPUResource<T>::CPUResource(T *data, unsigned int size, bool owner) :
_data(data), _size(size), _isOwner(owner), _isCPUResource(true) {
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
unsigned int CPUResource<T>::size() const {
	return _size;
}

template <typename T>
unsigned int CPUResource<T>::bytes() const {
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
ostream &operator<<(ostream &out, const CPUResource<T> &resource) {
	out << "::CPURessource::" << endl;
	out << "\t Is CPU Ressource : " << resource.isCPUResource() << endl;
	out << "\t Ressource type : " << resource.getResourceType() << endl;
	out << "\t Data : " << typeid(T).name() << endl;
	out << "\t Size : " << resource.size() << endl;
	out << "\t Bytes : " << resource.bytes() << endl;

	return out;
}
