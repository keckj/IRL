#ifndef CPURESSOURCE_H

#define CPURESSOURCE_H

#include <iostream>

template <typename T>
class CPUResource {
public:
	
	void setData(T* data, unsigned int size, bool isOwner);

	T* data() const;
	unsigned long size() const; //desired size
	unsigned long bytes() const; //desired bytes
	bool isOwner() const;
	
	bool isCPUResource() const; //is it allocated ?

	void setSize(unsigned int size);
	
	virtual void free() = 0;
	virtual const std::string getResourceType() const = 0;

protected:
	CPUResource(unsigned long size = 0); //wont be allocated
	CPUResource(CPUResource &original);
	CPUResource(T *data, unsigned int size, bool owner = false);
	virtual ~CPUResource();


	T* _data;
	unsigned long _size;
	bool _isOwner;
	bool _isCPUResource;
};

template <typename T>
std::ostream &operator<<(std::ostream &out, const CPUResource<T> &resource);

#include "CPUResource.tpp"

#endif /* end of include guard: CPURESSOURCE_H */
