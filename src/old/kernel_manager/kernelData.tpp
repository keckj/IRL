template <typename T>
KernelData<T>::KernelData(T *data, dim3 nData,
		DataType dataType, 
		DataRepresentation dataRepresentation,
		KernelData<T> *father,
		bool freeData)
: data(data), nData(nData),
	dataType(dataType), dataRepresentation(dataRepresentation),
	father(father), freeData(freeData)
{
	assert(nData.x>0 && nData.y>0 && nData.z>0);
}
			

template <typename T>
KernelData<T>::KernelData(const KernelData<T> &src) //copie tout sauf la data et met le père à null
: data(0), 
	nData(src.nData), 
	dataType(src.dataType), dataRepresentation(src.dataRepresentation),
	father(0), freeData(src.freeData)
{
}

template <typename T>
KernelData<T>::~KernelData() {
	if(freeData) {
		customFree<T>(dataType, data);
	}
}

template <typename T>
KernelData<T> ** KernelData<T>::split(SplitMethod sm, unsigned int n) const {
	assert(!(sm==LINEAR && n==0));
	assert(!(sm==MEMCPY && n==0));
	
	KernelData<T>** dataList = new KernelData<T>*[n];
	T* data;

	switch(sm) {
		case NOSPLIT:
			throw std::logic_error("Trying to split an unsplitable KernelData !");
			break;
		case MEMCPY:
			for (int i = 0; i < n; i++) {
				data = customMalloc(dataType, this->getDataNumber());
				data[i] = new KernelData(*this); //copy father params
				data[i]->setData(data).setFather(*this).setFree(true);
				customMemcpy(data[i], *this, this->getDataNumber());
			}
			break;
		case LINEAR:
			//for (int i = 0; i < n; i++) {
				//data = customMalloc(dataType, this->getDataNumber()/n);
				//data[i] = new KernelData(*this); //copy father params
				//data[i]->setData(data).setFather(*this).setFree(true);
			//}
			break;
		default:
			throw std::logic_error("Not implemented yet !");
	}
	return NULL;
}
			

template <typename T>
T* KernelData<T>::getData() const {
	return (this->data);
}

template <typename T>
unsigned int KernelData<T>::getDataNumber() const {
	return nData.x*nData.y*nData.z;
}

template <typename T>
unsigned int KernelData<T>::getDataType() const {
	return dataType;
}
			


template <typename T>
KernelData<T>& KernelData<T>::setData(T *data) {
	this->data = data;
	return *this;
}

template <typename T>
KernelData<T>& KernelData<T>::setSize(dim3 size) {
	this->size = size;
	return *this;
}

template <typename T>
KernelData<T>& KernelData<T>::setDataType(DataType newDataType) {
	this->dataType = newDataType;
	return *this;
}

template <typename T>
KernelData<T>& KernelData<T>::setDataRepresentation(DataRepresentation newDataRepresentation) {
	this->dataRepresentation = newDataRepresentation;
	return *this;
}

template <typename T>
KernelData<T>& KernelData<T>::setFather(KernelData<T> *father) {
	this->father = father;
	return *this;
}

template <typename T>
KernelData<T>& KernelData<T>::setFreeData(bool freeData) {
	this->freeData = freeData;
	return *this;
}


template <typename T>
T* customMalloc(DataType dataType, unsigned int nData) {

	T* ptr;

	switch(dataType) {
		case DEVICE_DATA_POINTER:
			cudaMalloc((void**) &ptr, nData);
			break;
		case HOST_PINNED_DATA_POINTER:
			cudaMallocHost((void**) &ptr, nData);
			break;
		case HOST_DATA_POINTER:
			ptr = new T[nData];
			break;
	}

	return ptr;
}

template <typename T>
void customFree(DataType dataType, T *ptr) {
	switch(dataType) {
		case DEVICE_DATA_POINTER:
			cudaFree(ptr);
			break;
		case HOST_PINNED_DATA_POINTER:
			cudaFreeHost(ptr);
			break;
		case HOST_DATA_POINTER:
			delete [] (ptr);
			break;
	}
}
	
template <typename T>
void customMemcpy(T *dst, DataType dstDataType, T *src, DataType srcDataType, 
		unsigned int nData, unsigned int srcOffset, unsigned int dstOffset) {
	
	switch(srcDataType) {
		
		case(HOST_DATA_POINTER):
		case(HOST_PINNED_DATA_POINTER):
			switch(dstDataType) {
				
				case(HOST_DATA_POINTER):
				case(HOST_PINNED_DATA_POINTER):
					cudaMemcpy(dst + dstOffset, src + srcOffset, nData*sizeof(T), cudaMemcpyHostToHost);
					break;
				case(DEVICE_DATA_POINTER):
					cudaMemcpy(dst + dstOffset, src + srcOffset, nData*sizeof(T), cudaMemcpyHostToDevice);
					break;
			}
			
			break;

		case(DEVICE_DATA_POINTER):
			switch(dstDataType) {
				
				case(HOST_DATA_POINTER):
				case(HOST_PINNED_DATA_POINTER):
					cudaMemcpy(dst + dstOffset, src + srcOffset, nData*sizeof(T), cudaMemcpyDeviceToHost);
					break;
				case(DEVICE_DATA_POINTER):
					cudaMemcpy(dst + dstOffset, src + srcOffset, nData*sizeof(T), cudaMemcpyDeviceToDevice);
					break;
			}

			break;
	}
}
	
template <typename T>
void customMemcpy(KernelData<T> dst, KernelData<T> src,
	unsigned int nData, unsigned int srcOffset = 0, unsigned int dstOffset = 0) {

	customMemcpy<T>(dst.getData(), dst.getDataType(), 
			src.getData(), src.getDataType(), 
			nData, dstOffset, srcOffset);
}
