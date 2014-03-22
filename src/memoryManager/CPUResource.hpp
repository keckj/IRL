#ifndef CPURESSOURCE_H

#define CPURESSOURCE_H

#include <iostream>

template <typename T>
class CPUResource {
public:
	

	void setData(T* data, unsigned int size, bool isOwner);

	T* data() const;
	unsigned int size() const;
	unsigned int bytes() const;
	bool isOwner() const;
	
	bool isCPUResource() const;
	
	virtual const std::string getResourceType() const = 0;

protected:
	CPUResource();
	CPUResource(T *data, unsigned int size, bool owner = false);
	virtual ~CPUResource();


	T* _data;
	unsigned int _size;
	bool _isOwner;
	bool _isCPUResource;
};

template <typename T>
std::ostream &operator<<(std::ostream &out, const CPUResource<T> &resource);

#include "CPUResource.tpp"

#endif /* end of include guard: CPURESSOURCE_H */
