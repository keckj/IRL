
#ifndef _KERNEL_MANAGER_TYPES_
#define _KERNEL_MANAGER_TYPES_

namespace kernel_manager {

	enum DataType { 
	DEVICE_DATA_POINTER, //DEVICE_SYMBOL,
	HOST_PINNED_DATA_POINTER, HOST_DATA_POINTER };

	enum DataRepresentation { ARRAY_1D, ARRAY_2D, ARRAY_3D };
	
	enum ResultOperation { BINARY_OPERATION, CONCATENATE, REDUCE };
	enum BinaryOperation { MIN, MAX, SUM, SUB, MULT, DIV, AVERAGE };

	enum SplitMethod {NOSPLIT, MEMCPY, LINEAR, QUADRANT, OCTANT };
}

#endif
