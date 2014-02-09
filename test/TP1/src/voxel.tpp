
#include "voxel.hpp"

template <class T>
Voxel<T>::Voxel(T value, bool active) : value(value), active(active) {
}

template <class T>
Voxel<T>::Voxel(Voxel<T> const & other) : value(0), active(other.active) {
	if(other.value) {
		this->value = T(other.value);
	}
}

template <class T>
Voxel<T>& Voxel<T>::operator=(const Voxel<T>& source) {
	
	Voxel tmp(source);

	std::swap(tmp.value, this->value);
	std::swap(tmp.active, this->active);
	
	return *this;
}

template <class T>
T Voxel<T>::getValue() {
	return this->value;
}

template <class T>
bool Voxel<T>::isActive() {
	return this->active;
}

template <class T>
void Voxel<T>::setActive(bool active) {
	this->active = active;
}

