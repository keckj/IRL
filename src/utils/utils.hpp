#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <sstream>

#include "cuda_runtime.h"

const std::string toStringMemory(unsigned long bytes);
const std::string toStringDim(dim3 d);
bool isPow2(unsigned int n);

#endif /* end of include guard: UTILS_H */
