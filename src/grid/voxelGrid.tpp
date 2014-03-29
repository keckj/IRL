
#include "voxelGrid.hpp"
#include <cmath>
#include <cassert>
#include <iostream>

template <typename T,
	template <typename T> class CPUResourceType,
	template <typename T> class GPUResourceType>
VoxelGrid<T,CPUResourceType,GPUResourceType>::VoxelGrid(
		const float gridRealWidth, 
		const float gridRealHeight, 
		const float gridRealLength, 
		const float deltaGrid,
		int deviceId) :
	SharedResource<T,CPUResourceType,GPUResourceType>(
			ceil(gridRealWidth/deltaGrid)
			* ceil(gridRealHeight/deltaGrid)
			* ceil(gridRealLength/deltaGrid),
			deviceId),
	_gridWidth(ceil(gridRealWidth/deltaGrid)),
	_gridHeight(ceil(gridRealHeight/deltaGrid)),
	_gridLength(ceil(gridRealLength/deltaGrid)),
	_deltaGrid(deltaGrid){
	}


template <typename T,
	template <typename T> class CPUResourceType,
	template <typename T> class GPUResourceType>
VoxelGrid<T,CPUResourceType,GPUResourceType>::VoxelGrid(
		const unsigned int gridWidth, 
		const unsigned int gridHeight, 
		const unsigned int gridLength, 
		const float deltaGrid,
		int deviceId) :
	SharedResource<T,CPUResourceType,GPUResourceType>(gridWidth*gridHeight*gridLength, deviceId),
	_gridWidth(gridWidth),
	_gridHeight(gridHeight),
	_gridLength(gridLength),
	_deltaGrid(deltaGrid)
	{
	}
	

template <typename T,
	template <typename T> class CPUResourceType,
	template <typename T> class GPUResourceType>
VoxelGrid<T,CPUResourceType,GPUResourceType>::VoxelGrid(VoxelGrid<T,CPUResourceType,GPUResource> &original) :
	SharedResource<T,CPUResourceType,GPUResourceType>(original),
	_gridWidth(original.width()),
	_gridHeight(original.height()),
	_gridLength(original.length()),
	_deltaGrid(original.voxelSize())
{
}
	
template <typename T,
	template <typename T> class CPUResourceType,
	template <typename T> class GPUResourceType>
VoxelGrid<T,CPUResourceType,GPUResourceType>::~VoxelGrid() {
}

template <typename T,
	template <typename T> class CPUResourceType,
	template <typename T> class GPUResourceType>
unsigned int VoxelGrid<T,CPUResourceType,GPUResourceType>::width() const {
	return this->_gridWidth;
}
template <typename T,
	template <typename T> class CPUResourceType,
	template <typename T> class GPUResourceType>
unsigned int VoxelGrid<T,CPUResourceType,GPUResourceType>::height() const {
	return this->_gridHeight;
}
template <typename T,
	template <typename T> class CPUResourceType,
	template <typename T> class GPUResourceType>
unsigned int VoxelGrid<T,CPUResourceType,GPUResourceType>::length() const {
	return this->_gridLength;
}
template <typename T,
	template <typename T> class CPUResourceType,
	template <typename T> class GPUResourceType>
float VoxelGrid<T,CPUResourceType,GPUResourceType>::voxelSize() const {
	return this->_deltaGrid;
}
	
template <typename T,
	template <typename T> class CPUResourceType,
	template <typename T> class GPUResourceType>
T VoxelGrid<T,CPUResourceType,GPUResourceType>::operator()(unsigned int i, unsigned int j, unsigned int k) {
	assert(i < _gridWidth);
	assert(j < _gridHeight);
	assert(k < _gridLength);
	
	return CPUResourceType<T>::_data[k*_gridWidth*_gridHeight + j*_gridWidth + i];
}
