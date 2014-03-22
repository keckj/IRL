
#include <cassert>
#include <unistd.h>

#include "CPUMemory.hpp"
#include "utils/utils.hpp"

const volatile unsigned long CPUMemory::_memorySize = 0;
const volatile unsigned long CPUMemory::_memoryRuntime = 0;
unsigned long CPUMemory::_memoryLeft = 0;

CPUMemory::CPUMemory() {
}

void CPUMemory::init() {
	
	long phypz = sysconf(_SC_PHYS_PAGES);
	long psize = sysconf(_SC_PAGE_SIZE);

	const_cast<unsigned long &>(CPUMemory::_memorySize) = phypz*psize;
	const_cast<unsigned long &>(CPUMemory::_memoryRuntime) = __CPU_MIN_RESERVED_MEMORY;
	CPUMemory::_memoryLeft = phypz*psize - __CPU_MIN_RESERVED_MEMORY;
}

unsigned long CPUMemory::memorySize() {
	return CPUMemory::_memorySize;
}

unsigned long CPUMemory::memoryLeft() {
	return CPUMemory::_memoryLeft;
}


void CPUMemory::display(std::ostream &out) {
	out << ":: CPU RAM Status ::" << std::endl; 
	out << "\t Total : " << toStringMemory(CPUMemory::_memorySize) 
		<< "\t Reserved : " << toStringMemory(CPUMemory::_memoryRuntime)
		<< "\t Used : " << toStringMemory(CPUMemory::_memorySize - CPUMemory::_memoryLeft) 
		<< "\t " << 100*(float)(CPUMemory::_memorySize - CPUMemory::_memoryLeft)/CPUMemory::_memorySize << "%"
		<< std::endl; 
}
