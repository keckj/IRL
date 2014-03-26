
#ifndef PINNEDCPURESOURCE_H
#define PINNEDCPURESOURCE_H

#include "CPUResource.hpp"

template <typename T>
class PinnedCPUResource : public CPUResource<T> {
	
public:
	PinnedCPUResource();
	PinnedCPUResource(T *data, unsigned int size, bool owner = false);
	~PinnedCPUResource();

	void free();

	const std::string getResourceType() const;
};

#include "PinnedCPUResource.tpp"


#endif /* end of include guard: PINNEDCPURESOURCE_H */
