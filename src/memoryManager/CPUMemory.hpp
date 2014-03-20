
#ifndef __CPU_MAX_MEMORY
#define __CPU_MAX_MEMORY 16000000
#endif

#ifndef CPUMEMORY_H
#define CPUMEMORY_H

#include "CPUResource.hpp"

class CPUMemory {
	public:
		static inline const unsigned long memorySize() const; //bytes
		static inline unsigned long remainingMemory() const; //bytes

		static inline bool canAllocate(unsigned long NBytes) const;

		static inline void *malloc(unsigned long NBytes, bool pagedMemory=false);
		static inline void free(CPUResource &src, bool force=false);

	private:
		CPUMemory();
		static const unsigned long memorySize;
		static unsigned long memoryLeft;
};

#endif /* end of include guard: CPUMEMORY_H */
