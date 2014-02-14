
#ifndef _CUDA_UTILS_
#define _CUDA_UTILS_

#include <ostream>
#include <log4cpp/Category.hh>
#include "cuda.h"
#include "cuda_runtime.h"

#include "utils/log.hpp"


#define CHECK_CUDA_ERRORS(ans) { gpuAssert((ans), __FILE__, __LINE__); }

class CudaUtils {

	public:
		static void printCudaDevices(std::ostream &outputStream);
		static void logCudaDevices(log4cpp::Category &log_output);

};

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true) {
	if (code != cudaSuccess) 
	{
		log_console.errorStream() << "GPU Assert => " << cudaGetErrorString(code) << " in file " <<  file << ":" << line << ".";
		if (abort) 
			throw std::logic_error("GPU Assert false !");
	}
}

#endif
