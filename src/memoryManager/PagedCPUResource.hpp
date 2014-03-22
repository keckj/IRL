

#ifndef PAGEDCPURESOURCE_H
#define PAGEDCPURESOURCE_H

#include "CPUResource.hpp"




template <typename T>
class PagedCPUResource : public CPUResource<T> {
	
public:
	PagedCPUResource();
	PagedCPUResource(T *data, unsigned int size, bool owner = false);
	~PagedCPUResource();

	const std::string getResourceType() const;
};

#include "PagedCPUResource.tpp"

#endif /* end of include guard: PAGEDCPURESOURCE_H */
