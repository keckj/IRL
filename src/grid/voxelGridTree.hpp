
#ifndef VOXELGRIDTREE_H
#define VOXELGRIDTREE_H

#include <vector>
#include "PowerOfTwoVoxelGrid.hpp"

template <typename T>
class VoxelGridTree {
	public:
		VoxelGridTree(unsigned int nGridX, unsigned int nGridY, unsigned int nGridZ,
			unsigned int subWidth, unsigned int subHeight, unsigned int subLength,
			std::vector<PowerOfTwoVoxelGrid<T> *> grids);

		std::vector<PowerOfTwoVoxelGrid<T> *>::iterator begin();
		std::vector<PowerOfTwoVoxelGrid<T> *>::const_iterator begin();

		std::vector<PowerOfTwoVoxelGrid<T> *>::iterator end();
		std::vector<PowerOfTwoVoxelGrid<T> *>::const_iterator end();

		T operator()(unsigned int i, unsigned int j, unsigned int k); //getVal
		//PowerOfTwoVoxelGrid<T>* operator(unsigned int i, unsigned int j, unsigned int k); //get subgrid containing

	protected:
		unsigned int _nChild;
		unsigned int _nGridX, _nGridY, _nGridZ;
		unsigned int _width, _height, _length;
		unsigned int _subWidth, _subHeight, _subLength;
		
		std::vector<PowerOfTwoVoxelGrid<T> *> _childs;	

};

#include "voxelGridTree.tpp"

#endif /* end of include guard: VOXELGRIDTREE_H */
