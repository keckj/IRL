#ifndef GPURESSOURCE_H
#define GPURESSOURCE_H

#include <iostream>

template <typename T>
class GPUResource {
public:
	
	GPUResource(int device = 0, unsigned long size = 0);
	GPUResource(GPUResource<T> &original);
	explicit GPUResource(T *data, int deviceId, unsigned int size, bool owner);
	~GPUResource();

	void setData(T* data, int deviceId, unsigned int size, bool isOwner);
	void free();
	void allocate();

	void setSize(unsigned long size);

	T* data() const;
	int deviceId() const;
	unsigned long size() const;
	unsigned long bytes() const;

	bool isOwner() const;
	bool isGPUResource() const;
	
	const std::string getResourceType() const;

protected:
	T* _data;
	int _deviceId;
	unsigned long _size;
	bool _isOwner;
	bool _isGPUResource;

};

template <typename T>
std::ostream &operator<<(std::ostream &out, const GPUResource<T> &resource);

#include "GPUResource.tpp"

#endif /* end of include guard: GPURESOURCE_H */
