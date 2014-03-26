
#include "powerOfTwoVoxelGrid.hpp"
#include "utils/utils.hpp"
#include <cmath>
#include <cassert>

template <typename T>
PowerOfTwoVoxelGrid<T>::PowerOfTwoVoxelGrid(VoxelGrid<T> &originalGrid) :
	VoxelGrid<T>(originalGrid),
	_powX(ceil(log2(originalGrid.width()))),
	_powY(ceil(log2(originalGrid.height()))),
	_powZ(ceil(log2(originalGrid.length())))
{
	this->VoxelGrid<T>::_gridWidth = 1<<_powX;
	this->VoxelGrid<T>::_gridHeight = 1<<_powY;
	this->VoxelGrid<T>::_gridLength = 1<<_powZ;
}


//Crée une grille de taille 2^powX x 2^powY x 2^powZ
template <typename T>
PowerOfTwoVoxelGrid<T>::PowerOfTwoVoxelGrid(unsigned int powX, unsigned int powY, unsigned int powZ, float deltaGrid) :
	VoxelGrid<T>(1u<<powX,1u<<powY,1u<<powZ,deltaGrid),
	_powX(powX), _powY(powY), _powZ(powZ)
{
}

//Decoupe la grille en sous grilles
template <typename T>
VoxelGridTree<T> PowerOfTwoVoxelGrid<T>::splitGrid(unsigned int splitX, unsigned int splitY, unsigned int splitZ) {
	assert(splitX <= _powX);
	assert(splitY <= _powY);
	assert(splitZ <= _powZ);
	assert(this->dataHost() == 0);
	assert(this->dataDevice() == 0);

	std::vector<PowerOfTwoVoxelGrid<T> *> subGrids;
	for (unsigned char i = 0; i < (1u << splitX); i++) {
		for (unsigned char j = 0; j < (1u << splitY); j++) {
			for (unsigned char k = 0; k < (1u << splitZ); k++) {
				subGrids.push_back(new PowerOfTwoVoxelGrid<T>(_powX-splitX, _powY-splitY, _powZ-splitZ, this->_deltaGrid));
			}
		}
	}

	return VoxelGridTree<T>(1<<splitX,1<<splitY,1<<splitZ, 
			subGrids[0]->width(), subGrids[0]->height(), subGrids[0]->length(),
			subGrids);
}

template <typename T>
VoxelGridTree<T> PowerOfTwoVoxelGrid<T>::splitGridWithMaxMemory(unsigned long maxMemoryPerSubgrid) {
	unsigned int pow = this->_powX + this->_powY + this->_powZ;
	unsigned int maxPow = floor(log2(maxMemoryPerSubgrid));
	unsigned int nSplit = ((int)pow - (int)maxPow >= 1 ? pow - maxPow : 1); //on split mini en deux

	//std::cout << "\n"<<nSplit << " "<<pow << " " << maxPow << " POW" << std::endl;
	assert(nSplit <= pow);

	unsigned int powX = this->_powX, powY = this->_powY, powZ = this->_powZ;
	unsigned int splitX=0, splitY=0, splitZ=0;

	for (unsigned int i = 0; i < nSplit; i++) {
		if(powX > powY && powX > powZ) { //on privilégie la découpe en z puis y puis x
			powX--;
			splitX++;
		}
		else {
			if(powY > powZ) {
				powY--;
				splitY++;
			}
			else {
				powZ--;
				splitZ++;
			}
		}
	}

	return this->splitGrid(splitX, splitY, splitZ);
}

template <typename T>
unsigned int PowerOfTwoVoxelGrid<T>::powX() {
	return _powX;
}

template <typename T>
unsigned int PowerOfTwoVoxelGrid<T>::powY() {
	return _powY;
}

template <typename T>
unsigned int PowerOfTwoVoxelGrid<T>::powZ() {
	return _powZ;
}
