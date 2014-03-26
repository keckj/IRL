#ifndef VOXELGRID_H
#define VOXELGRID_H

#include "memoryManager/PinnedCPUResource.hpp"
#include "memoryManager/GPUResource.hpp"

template <typename T>
class VoxelGrid : public PinnedCPUResource<T>, public GPUResource<T> {

public:
	explicit VoxelGrid(
			const float gridRealWidth, 
			const float gridRealHeight, 
			const float gridRealLength, 
			const float deltaGrid);
	
	explicit VoxelGrid(const unsigned int gridWidth, 
			const unsigned int gridHeight, 
			const unsigned int gridLength, 
			const float deltaGrid);
	
	virtual ~VoxelGrid();

	unsigned int width() const;
	unsigned int height() const;
	unsigned int length() const;
	float voxelSize() const;
	unsigned int dataSize() const;
	unsigned int dataBytes() const;

	unsigned char *dataHost() const;
	unsigned char *dataDevice() const;

	void allocateOnHost();
	void allocateOnDevice(int deviceId);

	unsigned char operator()(unsigned int i, unsigned int j, unsigned int k);

protected:
	const unsigned int _gridWidth, _gridHeight, _gridLength;
	const float _deltaGrid;
};

#include "voxelGrid.tpp"


#endif /* end of include guard: VOXELGRID_H */
