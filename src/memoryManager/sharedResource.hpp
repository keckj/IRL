
#ifndef SHAREDRESOURCE_H
#define SHAREDRESOURCE_H

#include "CPUResource.hpp"
#include "GPUResource.hpp"
#include <type_traits>


//wrapper class for CPU and GPU memory ressources
template <typename T, 
		 template <typename> class CPUResourceType, 
		 template <typename> class GPUResourceType>
		 class SharedResource : public CPUResourceType<T>, public GPUResourceType<T> {

			 static_assert(std::is_base_of<CPUResource<T>, CPUResourceType<T> >::value, "CPURessourceType<T> must extend CPURessource<T>");
			 static_assert(std::is_base_of<GPUResource<T>, GPUResourceType<T> >::value, "GPURessourceType<T> must extend GPURessource<T>");

			 public:
			 explicit SharedResource(int deviceId, unsigned long size);
			 SharedResource(GPUResourceType<T> gpuResource);
			 SharedResource(CPUResourceType<T> cpuResource, int deviceId);
			 SharedResource(CPUResourceType<T> cpuResource, GPUResourceType<T> gpuResource);
			 SharedResource(SharedResource<T, CPUResourceType, GPUResourceType> &original);

			 T* hostData() const;
			 T* deviceData() const;

			 void allocateOnHost();
			 void allocateOnDevice();
			 void allocateAll();

			 unsigned long dataSize() const;
			 unsigned long dataBytes() const;

			 int deviceId() const;

			 void copyToDevice(cudaStream_t stream = 0);
			 void copyToHost(cudaStream_t stream = 0);

		 };

#include "sharedResource.tpp"

#endif /* end of include guard: SHAREDRESOURCE_H */
