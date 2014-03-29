#ifndef VOXELGRID_H
#define VOXELGRID_H

#include "memoryManager/PinnedCPUResource.hpp"
#include "memoryManager/GPUResource.hpp"
#include "memoryManager/sharedResource.hpp"

template <typename T, 
		 template <typename T> class CPUResourceType, 
		 template <typename T> class GPUResourceType >
class VoxelGrid : public SharedResource<T, CPUResourceType, GPUResourceType> {

public:
	explicit VoxelGrid(
			float gridRealWidth, 
			float gridRealHeight, 
			float gridRealLength, 
			float deltaGrid,
			int deviceId = 0);
	
	explicit VoxelGrid(
			unsigned int gridWidth, 
			unsigned int gridHeight, 
			unsigned int gridLength, 
			float deltaGrid,
			int deviceId = 0);
	
	VoxelGrid(VoxelGrid<T,CPUResourceType,GPUResource> &original);

	virtual ~VoxelGrid();

	unsigned int width() const;
	unsigned int height() const;
	unsigned int length() const;
	float voxelSize() const;
	
	T operator()(unsigned int i, unsigned int j, unsigned int k);

protected:
	unsigned int _gridWidth, _gridHeight, _gridLength;
	float _deltaGrid;
};

#include "voxelGrid.tpp"


#endif /* end of include guard: VOXELGRID_H */
