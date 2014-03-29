
#ifndef POWEROFTWOVOXELGRID_H
#define POWEROFTWOVOXELGRID_H

#include "voxelGridTree.hpp"
#include "voxelGrid.hpp"

template <typename T, 
		 template <typename T> class CPUResourceType, 
		 template <typename T> class GPUResourceType >
class PowerOfTwoVoxelGrid : public VoxelGrid<T, CPUResourceType, GPUResourceType> {


	public:
		//Crée la grille enveloppante de taille puissance de 2 sur chaque axe
		explicit PowerOfTwoVoxelGrid(VoxelGrid<T, CPUResourceType, GPUResourceType> &originalGrid);

		//Crée une grille de taille 2^powX x 2^powY x 2^powZ
		explicit PowerOfTwoVoxelGrid(unsigned int powX, unsigned int powY, unsigned int powZ, float deltaGrid, int deviceId = 0);

		//Decoupe la grille en sous grilles 
		VoxelGridTree<T,CPUResourceType,GPUResourceType> splitGrid(unsigned int NSliceX, unsigned int NSliceY, unsigned int NSliceZ);

		//Decoupe la grille en fonction de la mémoire disponible
		//nb de split = max(minmemorySplit, 2^(minSplit))
		VoxelGridTree<T,CPUResourceType,GPUResourceType> splitGridWithMaxMemory(unsigned long maxMemoryPerSubgrid, unsigned int minSplits = 0); 

		unsigned int powX();
		unsigned int powY();
		unsigned int powZ();

	protected:
		unsigned int _powX, _powY, _powZ;
};

#include "powerOfTwoVoxelGrid.tpp"

#endif /* end of include guard: POWEROFTWOVOXELGRID_H */

