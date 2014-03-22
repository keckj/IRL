
#include <cassert>

#include "CPUMemory.hpp"
#include "utils/cudaUtils.hpp"

const unsigned long CPUMemory::_memorySize = __CPU_MAX_MEMORY;
unsigned long CPUMemory::_memoryLeft = __CPU_MAX_MEMORY;
		
CPUMemory::CPUMemory() {
}

unsigned long CPUMemory::memorySize() {
	return CPUMemory::_memorySize;
}

unsigned long CPUMemory::memoryLeft() {
	return CPUMemory::_memoryLeft;
}

		
