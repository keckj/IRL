
#ifndef POWEROFTWOVOXELGRID_H
#define POWEROFTWOVOXELGRID_H

#include "voxelGridTree.hpp"
#include "voxelGrid.hpp"


template <typename T>
class PowerOfTwoVoxelGrid : public VoxelGrid<T> {


	public:
		//Crée la grille enveloppante de taille puissance de 2 sur chaque axe
		explicit PowerOfTwoVoxelGrid(const VoxelGrid<T> &originalGrid);

		//Crée une grille de taille 2^powX x 2^powY x 2^powZ
		explicit PowerOfTwoVoxelGrid(unsigned char powX, unsigned char powY, unsigned char powZ, const float deltaGrid);

		//Decoupe la grille en sous grilles 
		VoxelGridTree<T> *splitGrid(unsigned char NSliceX, unsigned char NSliceY, unsigned char NSliceZ);

		unsigned char powX();
		unsigned char powY();
		unsigned char powZ();

	protected:
		unsigned char _powX, _powY, _powZ;
};

#include "PowerOfTwoVoxelGrid.tpp"

#endif /* end of include guard: POWEROFTWOVOXELGRID_H */

