
#ifndef POWEROFTWOVOXELGRID_H
#define POWEROFTWOVOXELGRID_H

class PowerOfTwoVoxelGrid : public VoxelGrid {


	public:
		//Crée la grille enveloppante de taille puissance de 2 sur chaque axe
		explicit PowerOfTwoVoxelGrid(const VoxelGrid &originalGrid)

		//Crée une grille de taille 2^powX x 2^powY x 2^powZ
		explicit PowerOfTwoVoxelGrid(unsigned char powX, unsigned char powY, unsigned char powZ);

		//Decoupe la grille en sous grilles 
		VoxelGridPack *splitGrid(unsigned char NSliceX, unsigned char NSliceY, unsigned char NSliceZ);

	protected:
		unsigned char powX, powY, powZ;
};

#endif /* end of include guard: POWEROFTWOVOXELGRID_H */

