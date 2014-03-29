
#include "utils.hpp"
#include <cmath>

const std::string toStringMemory(unsigned long bytes) {

	std::stringstream ss;

	const char prefix[] = {' ', 'K', 'M', 'G', 'T', 'P'};
	unsigned long val = 1;
	for (int i = 0; i < 6; i++) {
		if(bytes < 1024*val) {
			ss << round(100*(float)bytes/val)/100.0 << prefix[i] << 'B';
			break;
		}
		val *= 1024;
	}

	const std::string str(ss.str());
	return str;
}

bool isPow2(unsigned int n) {
	
	while(n>1) {
		if(n%2==1)
			return false;
		n = n/2;
	}

	return true;
}

const std::string toStringDim(dim3 d) {
	std::stringstream ss;
	ss << "(" << d.x << "," << d.y << "," << d.z << ")";
	const std::string str(ss.str());
	return str;
}


