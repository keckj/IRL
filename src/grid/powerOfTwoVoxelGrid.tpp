
#include "PowerOfTwoVoxelGrid.hpp"
#include "utils/utils.hpp"
#include <cmath>
#include <cassert>

template <typename T>
PowerOfTwoVoxelGrid<T>::PowerOfTwoVoxelGrid(const VoxelGrid<T> &originalGrid) :
	VoxelGrid<T>(originalGrid)
{
	_powX = ceil(log2(originalGrid.width()));
	_powY = ceil(log2(originalGrid.height()));
	_powZ = ceil(log2(originalGrid.length()));
	this->gridWidth = 1 << _powX;
	this->gridHeight = 1 << _powY;
	this->gridLength = 1 << _powZ;
}


//Crée une grille de taille 2^powX x 2^powY x 2^powZ
template <typename T>
PowerOfTwoVoxelGrid<T>::PowerOfTwoVoxelGrid(unsigned char powX, unsigned char powY, unsigned char powZ, const float deltaGrid) :
VoxelGrid<T>(PinnedCPUResource<T>(), GPUResource<T>(), 1<<powX,1<<powY,1<<powZ,deltaGrid),
_powX(powX), _powY(powY), _powZ(powZ)
{
}

//Decoupe la grille en sous grilles 
template <typename T>
VoxelGridTree<T> *PowerOfTwoVoxelGrid<T>::splitGrid(unsigned char NSliceX, unsigned char NSliceY, unsigned char NSliceZ) {
	assert(isPow2(NSliceX));
	assert(isPow2(NSliceY));
	assert(isPow2(NSliceZ));
	assert(NSliceX >= powX);
	assert(NSliceY >= powY);
	assert(NSliceZ >= powZ);
	assert(this->dataHost() == 0);
	assert(this->dataDevice() == 0);
	
	std::vector<PowerOfTwoVoxelGrid<T> *> subGrids;
	for (unsigned char i = 0; i < NSliceX; i++) {
		for (unsigned char j = 0; j < NSliceY; j++) {
			for (unsigned char k = 0; k < NSliceZ; k++) {
				subGrids->push_back(new PowerOfTwoVoxelGrid<T>(_powX-NSliceX, _powY-NSliceY, _powZ-NSliceZ, this->deltaGrid));
			}
		}
	}

	return new VoxelGridTree<T>(NSliceX,NSliceY,NSliceZ, 
			subGrids[0]->width(), subGrids[0]->height(), subGrids[0]->length(),
			subGrids);
}

template <typename T>
unsigned char PowerOfTwoVoxelGrid<T>::powX() {
	return _powX;
}

template <typename T>
unsigned char PowerOfTwoVoxelGrid<T>::powX() {
	return _powY;
}

template <typename T>
unsigned char PowerOfTwoVoxelGrid<T>::powX() {
	return _powZ;
}
