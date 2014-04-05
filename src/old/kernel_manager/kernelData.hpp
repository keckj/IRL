

#ifndef _KERNEL_MANAGER_KERNEL_DATA__
#define _KERNEL_MANAGER_KERNEL_DATA_

#include "types.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <cassert>


namespace kernel_manager {

	template <typename T>
	class KernelData {
		
		public:
			KernelData(T *data, dim3 nData, 
					DataType dataType, 
					DataRepresentation dataRepresentation = ARRAY_1D,
					KernelData<T> *father = 0,
					bool freeData = true);

			KernelData(const KernelData &src); //copie tout les attributs sauf data et father (NULL)

			~KernelData();

			KernelData<T>& setData(T *data);
			KernelData<T>& setSize(dim3 size);
			KernelData<T>& setDataType(DataType newDataType);
			KernelData<T>& setDataRepresentation(DataRepresentation newDataRepresentation);
			KernelData<T>& setFather(KernelData<T> *father);
			KernelData<T>& setFreeData(bool freeData);
			
			KernelData<T> ** split(SplitMethod sm, unsigned int n) const;

			T* getData() const;
			unsigned int getDataNumber() const;
			unsigned int getDataType() const;
			
			
		private:
			T *data;
			
			dim3 nData;
			unsigned int elementSize;
			
			DataType dataType;
			DataRepresentation dataRepresentation;

			KernelData<T> *father;
			bool freeData;
	};

	
	template <typename T>
	static T* customMalloc(DataType dataType, unsigned int nData);

	template <typename T>
	static void customFree(DataType dataType, T *ptr);
	
	template <typename T>
	static void customMemcpy(T *dst, DataType dstDataType, T *src, DataType srcDataType, 
		unsigned int ndata, unsigned int srcOffset = 0, unsigned int dstOffset = 0);
	
	template <typename T>
	static void customMemcpy(KernelData<T> dst, KernelData<T> src,
		unsigned int ndata, unsigned int srcOffset = 0, unsigned int dstOffset = 0);
	
	#include "kernelData.tpp"
}

#endif
