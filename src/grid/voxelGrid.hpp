#ifndef VOXELGRID_H
#define VOXELGRID_H

#include "memoryManager/PinnedCPUResource.hpp"
#include "memoryManager/GPUResource.hpp"

template <typename T>
class VoxelGrid : public PinnedCPUResource<T>, public GPUResource<T> {

public:
	explicit VoxelGrid(
			float gridRealWidth, 
			float gridRealHeight, 
			float gridRealLength, 
			float deltaGrid);
	
	explicit VoxelGrid(unsigned int gridWidth, 
			unsigned int gridHeight, 
			unsigned int gridLength, 
			float deltaGrid);
	
	virtual ~VoxelGrid();

	unsigned int width() const;
	unsigned int height() const;
	unsigned int length() const;
	float voxelSize() const;
	
	unsigned long dataSize() const;
	unsigned long dataBytes() const;

	T *dataHost() const;
	T *dataDevice() const;

	void allocateOnHost();
	void allocateOnDevice(int deviceId);

	T operator()(unsigned int i, unsigned int j, unsigned int k);

protected:
	unsigned int _gridWidth, _gridHeight, _gridLength;
	float _deltaGrid;
};

#include "voxelGrid.tpp"


#endif /* end of include guard: VOXELGRID_H */
