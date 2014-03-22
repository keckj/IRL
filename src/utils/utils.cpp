
#include "utils.hpp"

const std::string toStringMemory(unsigned long bytes) {

	std::stringstream ss;
	
	if(bytes < 1000) {
		ss << bytes << "B";
	}
	else if(bytes < 1000000) {
		ss << bytes/1000 << "KB";
	}
	else if(bytes < 1000000000) {
		ss << bytes/1000000 << "MB";
	}
	else if(bytes < 1000000000000) {
		ss << bytes/1000000000 << "GB";
	}
	else {
		ss << bytes/1000000000000 << "TB";
	}

	const std::string str(ss.str());
	return str;
}
