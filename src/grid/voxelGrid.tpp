
#include "voxelGrid.hpp"
#include <cmath>
#include <cassert>
#include <iostream>

template <typename T>
VoxelGrid<T>::VoxelGrid(
		const float gridRealWidth, 
		const float gridRealHeight, 
		const float gridRealLength, 
		const float deltaGrid) :
	PinnedCPUResource<T>(),
	GPUResource<T>(),
	_gridWidth(ceil(gridRealWidth/deltaGrid)),
	_gridHeight(ceil(gridRealHeight/deltaGrid)),
	_gridLength(ceil(gridRealLength/deltaGrid)),
	_deltaGrid(deltaGrid){
	}

template <typename T>
VoxelGrid<T>::VoxelGrid(
		const unsigned int gridWidth, 
		const unsigned int gridHeight, 
		const unsigned int gridLength, 
		const float deltaGrid) :
	PinnedCPUResource<T>(),
	GPUResource<T>(),
	_gridWidth(gridWidth),
	_gridHeight(gridHeight),
	_gridLength(gridLength),
	_deltaGrid(deltaGrid){
	}
	
template <typename T>
VoxelGrid<T>::~VoxelGrid() {
}

template <typename T>
void VoxelGrid<T>::allocateOnHost() {
	assert(CPUMemory::canAllocate<T>(this->dataSize()));
	T *data = CPUMemory::malloc<T>(this->dataSize(), true);	
	this->PinnedCPUResource<T>::setData(data, this->dataSize(), true);
}

template <typename T>
void VoxelGrid<T>::allocateOnDevice(int deviceId) {
	assert(GPUMemory::canAllocate<T>(this->dataSize(), deviceId));
	T *data = GPUMemory::malloc<T>(this->dataSize(), deviceId);	
	this->GPUResource<T>::setData(data, deviceId, this->dataSize(), true);
}


template <typename T>
unsigned int VoxelGrid<T>::width() const {
	return this->_gridWidth;
}
template <typename T>
unsigned int VoxelGrid<T>::height() const {
	return this->_gridHeight;
}
template <typename T>
unsigned int VoxelGrid<T>::length() const {
	return this->_gridLength;
}
template <typename T>
float VoxelGrid<T>::voxelSize() const {
	return this->_deltaGrid;
}
template <typename T>
unsigned int VoxelGrid<T>::dataSize() const {
	return this->_gridWidth * this->_gridHeight * this->_gridLength;
}
	
template <typename T>
unsigned int VoxelGrid<T>::dataBytes() const {
	return this->dataSize()*sizeof(T);	
}

template <typename T>
unsigned char *VoxelGrid<T>::dataHost() const {
	return PinnedCPUResource<T>::_data;
}
template <typename T>
unsigned char *VoxelGrid<T>::dataDevice() const {
	return GPUResource<T>::_data;
}
	
template <typename T>
unsigned char VoxelGrid<T>::operator()(unsigned int i, unsigned int j, unsigned int k) {
	assert(i < _gridWidth);
	assert(j < _gridHeight);
	assert(k < _gridLength);
	
	return PinnedCPUResource<T>::_data[k*_gridWidth*_gridHeight + j*_gridWidth + i];
}
