
#ifndef VOXELGRIDTREE_H
#define VOXELGRIDTREE_H

#include <vector>

class VoxelGridTree : {
public:
	VoxelGridTree(unsigned int NChild, std::list<PowerOfTwoVoxelGrid *> grids);

	unsigned int getNChild();
	PowerOfTwoVoxelGrid* getChild(unsigned int childId);

	std::vector<PowerOfTwoPowerOfTwoVoxelGrid *>::iterator begin();
	std::vector<PowerOfTwoPowerOfTwoVoxelGrid *>::const_iterator begin();
	
	std::vector<PowerOfTwoPowerOfTwoVoxelGrid *>::iterator end();
	std::vector<PowerOfTwoPowerOfTwoVoxelGrid *>::const_iterator end();

	unsigned char operator()(unsigned int i, unsigned int j, unsigned int k);
	signed int getChildID(unsigned int i, unsigned int j, unsigned int k);

protected:
	unsigned int width, height, length;
	unsigned char powX, powY, powZ;

	unsigned int NChild;
	std::vector<PowerOfTwoVoxelGrid *> childs;	

};


#endif /* end of include guard: VOXELGRIDTREE_H */
