
template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 VoxelGridTree<T,CPUResourceType,GPUResourceType>::VoxelGridTree(unsigned int nGridX, unsigned int nGridY, unsigned int nGridZ,
				 unsigned int subWidth, unsigned int subHeight, unsigned int subLength, float deltaGrid,
				 std::vector<PowerOfTwoVoxelGrid<T,CPUResourceType,GPUResourceType> *> grids) :
			 _nChild(nGridX*nGridY*nGridZ),
			 _nGridX(nGridX), _nGridY(nGridY), _nGridZ(nGridZ),
			 _width(nGridX*subWidth), _height(nGridY*subHeight), _length(nGridZ*subLength),
			 _subWidth(subWidth), _subHeight(subHeight), _subLength(subLength),
			 _deltaGrid(deltaGrid), 
			 _childs(grids)
{
}

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 typename std::vector<PowerOfTwoVoxelGrid<T,CPUResourceType,GPUResourceType> *>::iterator VoxelGridTree<T,CPUResourceType,GPUResourceType>::begin() {
			 return _childs.begin();
		 }

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 typename std::vector<PowerOfTwoVoxelGrid<T,CPUResourceType,GPUResourceType> *>::const_iterator VoxelGridTree<T,CPUResourceType,GPUResourceType>::cbegin() {
			 return _childs.begin();
		 }

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 typename std::vector<PowerOfTwoVoxelGrid<T,CPUResourceType,GPUResourceType> *>::iterator VoxelGridTree<T,CPUResourceType,GPUResourceType>::end() {
			 return _childs.end();
		 }

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 typename std::vector<PowerOfTwoVoxelGrid<T,CPUResourceType,GPUResourceType> *>::const_iterator VoxelGridTree<T,CPUResourceType,GPUResourceType>::cend() {
			 return _childs.end();
		 }

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 T VoxelGridTree<T,CPUResourceType,GPUResourceType>::operator()(unsigned int i, unsigned int j, unsigned int k) {

			 unsigned int nx = i/_subWidth;
			 unsigned int ny = j/_subHeight;
			 unsigned int nz = k/_subLength;
			 unsigned int n = nz*_nGridX*_nGridY + ny*_nGridX + nx;

			 unsigned int x = i % _subWidth;
			 unsigned int y = j % _subHeight;
			 unsigned int z = k % _subLength;

			 assert(n < _nChild);

			 return (*_childs[n])(x,y,z);
		 }

//get subgrid i 
template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 PowerOfTwoVoxelGrid<T,CPUResourceType,GPUResourceType>* 
		 VoxelGridTree<T,CPUResourceType,GPUResourceType>::operator()(unsigned int i) {
			 return _childs[i];
		 }

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 unsigned int VoxelGridTree<T,CPUResourceType,GPUResourceType>::nChilds() const {
			 return _nChild;
		 }

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 unsigned int VoxelGridTree<T,CPUResourceType,GPUResourceType>::width() const {
			 return this->_width;
		 }

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 unsigned int VoxelGridTree<T,CPUResourceType,GPUResourceType>::height() const {
			 return this->_height;
		 }

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 unsigned int VoxelGridTree<T,CPUResourceType,GPUResourceType>::length() const {
			 return this->_length;
		 }

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 unsigned int VoxelGridTree<T,CPUResourceType,GPUResourceType>::subwidth() const {
			 return this->_subWidth;
		 }

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 unsigned int VoxelGridTree<T,CPUResourceType,GPUResourceType>::subheight() const {
			 return this->_subHeight;
		 }

template <typename T,
template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 unsigned int VoxelGridTree<T,CPUResourceType,GPUResourceType>::sublength() const {
			 return this->_subLength;
		 }

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 float VoxelGridTree<T,CPUResourceType,GPUResourceType>::voxelSize() const {
			 return this->_deltaGrid;
		 }

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 unsigned long VoxelGridTree<T,CPUResourceType,GPUResourceType>::subgridSize() const {
			 return _subWidth * _subHeight * _subLength;
		 }

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 unsigned long VoxelGridTree<T,CPUResourceType,GPUResourceType>::subgridBytes() const {
			 return this->subgridSize() * sizeof(T);
		 }
		
template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 unsigned int VoxelGridTree<T,CPUResourceType,GPUResourceType>::nGridX() const {
			 return this->_nGridX;
	}

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 unsigned int VoxelGridTree<T,CPUResourceType,GPUResourceType>::nGridY() const {
			 return this->_nGridY;
	}

template <typename T,
		 template <typename> class CPUResourceType,
		 template <typename> class GPUResourceType>
		 unsigned int VoxelGridTree<T,CPUResourceType,GPUResourceType>::nGridZ() const {
			 return this->_nGridZ;
	}
