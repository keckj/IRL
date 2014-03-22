
#ifndef __CPU_MAX_MEMORY
#define __CPU_MAX_MEMORY 16000000
#endif

#ifndef CPUMEMORY_H
#define CPUMEMORY_H

#include "CPUResource.hpp"

class CPUMemory {
	public:
		static unsigned long memorySize(); //bytes
		static unsigned long memoryLeft(); //bytes

		static bool canAllocateBytes(unsigned int nBytes);

		template <typename T>
		static bool canAllocate(unsigned int nData);

		template <typename T>
		static T *malloc(unsigned int nData, bool pinnedMemory=false);

		template <typename T>
		static void free(T *data, unsigned int nData, bool force=false);
			

	private:
		CPUMemory();
		static const unsigned long _memorySize;
		static unsigned long _memoryLeft;
};

#include "CPUMemory.tpp"

#endif /* end of include guard: CPUMEMORY_H */
