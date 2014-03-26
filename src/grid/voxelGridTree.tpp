
template <typename T>
VoxelGridTree<T>::VoxelGridTree(unsigned int nGridX, unsigned int nGridY, unsigned int nGridZ,
		unsigned int subWidth, unsigned int subHeight, unsigned int subLength,
		std::vector<PowerOfTwoVoxelGrid<T> *> grids) :
_nChild(nGridX*nGridY*nGridZ),
_nGridX(nGridX), _nGridY(nGridY), _nGridZ(nGridZ),
_width(nGridX*subWidth), _height(nGridY*subHeight), _length(nGridZ*subLength),
_subWidth(subWidth), _subHeight(subHeight), _subLength(subLength),
_childs(grids)
{
}

template <typename T>
std::vector<PowerOfTwoVoxelGrid<T> *>::iterator VoxelGridTree<T>::begin() {
	return _childs.begin();
}

template <typename T>
std::vector<PowerOfTwoVoxelGrid<T> *>::const_iterator VoxelGridTree<T>::begin() {
	return _childs.begin();
}

template <typename T>
std::vector<PowerOfTwoVoxelGrid<T> *>::iterator VoxelGridTree<T>::end() {
	return _childs.end();
}

template <typename T>
std::vector<PowerOfTwoVoxelGrid<T> *>::const_iterator VoxelGridTree<T>::end() {
	return _childs.end();
}

template <typename T>
T VoxelGridTree<T>::operator()(unsigned int i, unsigned int j, unsigned int k) {
	
	unsigned int nx = i/_subWidth;
	unsigned int ny = j/_subHeight;
	unsigned int nz = k/_subLength;
	unsigned int n = nz*_nGridX*_nGridY + ny*_nGridX + nx;

	unsigned int x = i % _subWidth;
	unsigned int y = j % _subHeight;
	unsigned int z = k % _subLength;

	assert(n < _nChild);

	return _childs[i](x,y,z);
}
