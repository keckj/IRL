
#ifndef POWEROFTWOVOXELGRID_H
#define POWEROFTWOVOXELGRID_H

#include "voxelGridTree.hpp"
#include "voxelGrid.hpp"

template <typename T>
class PowerOfTwoVoxelGrid : public VoxelGrid<T> {


	public:
		//Crée la grille enveloppante de taille puissance de 2 sur chaque axe
		explicit PowerOfTwoVoxelGrid(VoxelGrid<T> &originalGrid);

		//Crée une grille de taille 2^powX x 2^powY x 2^powZ
		explicit PowerOfTwoVoxelGrid(unsigned int powX, unsigned int powY, unsigned int powZ, float deltaGrid);

		//Decoupe la grille en sous grilles 
		VoxelGridTree<T> splitGrid(unsigned int NSliceX, unsigned int NSliceY, unsigned int NSliceZ);

		//Decoupe la grille en fonction de la mémoire disponible
		VoxelGridTree<T> splitGridWithMaxMemory(unsigned long maxMemoryPerSubgrid);

		unsigned int powX();
		unsigned int powY();
		unsigned int powZ();

	protected:
		unsigned int _powX, _powY, _powZ;
};

#include "powerOfTwoVoxelGrid.tpp"

#endif /* end of include guard: POWEROFTWOVOXELGRID_H */

