#ifndef GPURESSOURCE_H
#define GPURESSOURCE_H

#include <iostream>

template <typename T>
class GPUResource {
public:
	
	GPUResource();
	GPUResource(GPUResource<T> &original);
	explicit GPUResource(T *data, int deviceId, unsigned int size, bool owner);
	~GPUResource();

	void setData(T* data, int deviceId, unsigned int size, bool isOwner);
	void free();

	T* data() const;
	int deviceId() const;
	unsigned int size() const;
	unsigned int bytes() const;

	bool isOwner() const;
	bool isGPUResource() const;
	
	const std::string getResourceType() const;

protected:
	T* _data;
	int _deviceId;
	unsigned int _size;
	bool _isOwner;
	bool _isGPUResource;

};

template <typename T>
std::ostream &operator<<(std::ostream &out, const GPUResource<T> &resource);

#include "GPUResource.tpp"

#endif /* end of include guard: GPURESOURCE_H */
