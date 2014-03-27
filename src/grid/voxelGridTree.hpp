
#ifndef VOXELGRIDTREE_H
#define VOXELGRIDTREE_H

#include <vector>
#include "powerOfTwoVoxelGrid.hpp"

template <typename T> class PowerOfTwoVoxelGrid;

template <typename T>
class VoxelGridTree {
	public:
		VoxelGridTree(unsigned int nGridX, unsigned int nGridY, unsigned int nGridZ,
			unsigned int subWidth, unsigned int subHeight, unsigned int subLength,
			std::vector<PowerOfTwoVoxelGrid<T> *> grids);

		typename std::vector<PowerOfTwoVoxelGrid<T> *>::iterator begin();
		typename std::vector<PowerOfTwoVoxelGrid<T> *>::const_iterator cbegin();

		typename std::vector<PowerOfTwoVoxelGrid<T> *>::iterator end();
		typename std::vector<PowerOfTwoVoxelGrid<T> *>::const_iterator cend();

		T operator()(unsigned int i, unsigned int j, unsigned int k); //getVal
		//PowerOfTwoVoxelGrid<T>* operator(unsigned int i, unsigned int j, unsigned int k); //get subgrid containing
		//
		unsigned int nChilds() const;
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
		
		std::vector<PowerOfTwoVoxelGrid<T> *> _childs;	

};

#include "voxelGridTree.tpp"

#endif /* end of include guard: VOXELGRIDTREE_H */
