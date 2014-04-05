
#ifndef VOXELGRIDTREE_H
#define VOXELGRIDTREE_H

#include <vector>
#include "powerOfTwoVoxelGrid.hpp"

template <typename T, 
		 template <typename> class CPUResourceType, 
		 template <typename> class GPUResourceType >
class PowerOfTwoVoxelGrid;

template <typename T, 
		 template <typename> class CPUResourceType, 
		 template <typename> class GPUResourceType >
class VoxelGridTree {
	public:
		VoxelGridTree(unsigned int nGridX, unsigned int nGridY, unsigned int nGridZ,
			unsigned int subWidth, unsigned int subHeight, unsigned int subLength,
			float deltaGrid,
			std::vector<PowerOfTwoVoxelGrid<T,CPUResourceType,GPUResourceType> *> grids);

		typename std::vector<PowerOfTwoVoxelGrid<T,CPUResourceType,GPUResourceType> *>::iterator begin();
		typename std::vector<PowerOfTwoVoxelGrid<T,CPUResourceType,GPUResourceType> *>::const_iterator cbegin();

		typename std::vector<PowerOfTwoVoxelGrid<T,CPUResourceType,GPUResourceType> *>::iterator end();
		typename std::vector<PowerOfTwoVoxelGrid<T,CPUResourceType,GPUResourceType> *>::const_iterator cend();

		T operator()(unsigned int i, unsigned int j, unsigned int k); //getVal
		PowerOfTwoVoxelGrid<T,CPUResourceType,GPUResourceType>* operator()(unsigned int i); //get subgrid i 

		unsigned int nChilds() const;

		unsigned int width() const;
		unsigned int height() const;
		unsigned int length() const;

		unsigned int subwidth() const;
		unsigned int subheight() const;
		unsigned int sublength() const;
		float voxelSize() const;

		unsigned long subgridSize() const;
		unsigned long subgridBytes() const;
		

	protected:
		unsigned int _nChild;
		unsigned int _nGridX, _nGridY, _nGridZ;
		unsigned int _width, _height, _length;
		unsigned int _subWidth, _subHeight, _subLength;

		float _deltaGrid;

		std::vector<PowerOfTwoVoxelGrid<T,CPUResourceType,GPUResourceType> *> _childs;	

};

#include "voxelGridTree.tpp"

#endif /* end of include guard: VOXELGRIDTREE_H */
