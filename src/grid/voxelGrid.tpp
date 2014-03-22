
#include "voxelGrid.hpp"
#include <cmath>
#include <cassert>
#include <iostream>

template <typename T>
VoxelGrid<T>::VoxelGrid(PinnedCPUResource<T> hostGrid,
		GPUResource<T> deviceGrid, 
		const float gridRealWidth, 
		const float gridRealHeight, 
		const float gridRealLength, 
		const float deltaGrid) :
	PinnedCPUResource<T>(hostGrid),
	GPUResource<T>(deviceGrid),
	gridWidth(ceil(gridRealWidth/deltaGrid)),
	gridHeight(ceil(gridRealHeight/deltaGrid)),
	gridLength(ceil(gridRealLength/deltaGrid)),
	deltaGrid(deltaGrid){
	}

template <typename T>
VoxelGrid<T>::VoxelGrid(PinnedCPUResource<T> hostGrid,
		GPUResource<T> deviceGrid, 
		const unsigned int gridWidth, 
		const unsigned int gridHeight, 
		const unsigned int gridLength, 
		const float deltaGrid) :
	PinnedCPUResource<T>(hostGrid),
	GPUResource<T>(deviceGrid),
	gridWidth(gridWidth),
	gridHeight(gridHeight),
	gridLength(gridLength),
	deltaGrid(deltaGrid){
	}

template <typename T>
VoxelGrid<T>::~VoxelGrid() {
}

template <typename T>
unsigned int VoxelGrid<T>::width() const {
	return this->gridWidth;
}
template <typename T>
unsigned int VoxelGrid<T>::height() const {
	return this->gridHeight;
}
template <typename T>
unsigned int VoxelGrid<T>::length() const {
	return this->gridLength;
}
template <typename T>
float VoxelGrid<T>::voxelSize() const {
	return this->deltaGrid;
}
template <typename T>
unsigned int VoxelGrid<T>::dataSize() const {
	return this->gridWidth * this->gridHeight * this->gridLength;
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
	assert(i < gridWidth);
	assert(j < gridHeight);
	assert(k < gridLength);
	
	return PinnedCPUResource<T>::_data[k*gridWidth*gridHeight + j*gridWidth + i];
}
