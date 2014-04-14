#include <cmath>

#include "utils/cudaUtils.hpp"
#include "utils/log.hpp"
#include "utils/utils.hpp"

namespace kernel {

__global__ void cast(const long dataSize, float *float_data, unsigned char *char_data) {
	
	unsigned int id = blockIdx.y*65535*512 + blockIdx.x*512 + threadIdx.x;

	if(id > dataSize)
		return;
	
	char_data[id] = (unsigned char) float_data[id];
}

void castKernel(unsigned long dataSize, float *float_data, unsigned char *char_data) {
	dim3 dimBlock(512);
	dim3 dimGrid((dataSize/512) % 65535, ceil(dataSize/(512*65535.0f)));
	log_console.infoStream() << "[KERNEL::Cast] <<<" << toStringDim(dimBlock) << ", " << toStringDim(dimGrid) << ">>>";
	cast<<<dimGrid,dimBlock>>>(dataSize, float_data, char_data);
}



__device__ double atomicadd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomiccas(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}




 void VNN(const int nImages, const int imgWidth, const int imgHeight, 
		const float deltaGrid, const float deltaX, const float deltaY,
		const float xMin, const float yMin, const float zMin,
		const unsigned int gridIdx, const unsigned int gridIdy, const unsigned int gridIdz,
		const unsigned int voxelGridWidth, const unsigned int voxelGridHeight, const unsigned int voxelGridLength,
		float *offsetX, float *offsetY, float *offsetZ,
		float *r1, float *r2, float *r3, float *r4, float *r5, float *r6, float *r7, float *r8, float *r9,
		unsigned char *char_image_data, 
		unsigned char *voxel_data, 
		unsigned short *mean_grid,
		unsigned char *hit_counter) {
	
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;  
	unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y; 
	unsigned int id = idy*imgWidth + idx;
	unsigned int n = blockIdx.z;

	if(idx >= imgWidth || idy >= imgHeight)
		return;

	float vx = (idx+0.5f)*deltaX;	
	float vy = (idy+0.5f)*deltaY;
	
	float px = r1[n]*vx + r2[n]*vy + r3[n]*0.0f + offsetX[n] - xMin;
	float py = r4[n]*vx + r5[n]*vy + r6[n]*0.0f + offsetY[n] - yMin;
	float pz = r7[n]*vx + r8[n]*vy + r9[n]*0.0f + offsetZ[n] - zMin;

	unsigned int ix = __float2uint_rd(px/deltaGrid);
	unsigned int iy = __float2uint_rd(py/deltaGrid);
	unsigned int iz = __float2uint_rd(pz/deltaGrid);
	
	//check if pixel is in the subgrid
	if(! (ix/voxelGridWidth == gridIdx 
		&& iy/voxelGridHeight == gridIdy
		&& iz/voxelGridLength == gridIdz)) {
		return;
	}
	
	ix %= voxelGridWidth;
	iy %= voxelGridHeight;
	iz %= voxelGridLength;
	unsigned long i = iz*voxelGridHeight*voxelGridWidth + iy*voxelGridWidth + ix;
	
	if(hit_counter[i] >= 255 - warpSize)
		return;

	unsigned char value = char_image_data[id];

	__shared__ unsigned int mutex;
	mutex = 0;
	__syncthreads();

	while(atomicCAS(&mutex, 0, 1) != 1);
	
	printf("thread passed !");
	if(hit_counter[i] == 255)
		return;

	hit_counter[i]++;	
	mean_grid[i]+=value;

	mutex = 0;

	__syncthreads();
	if(hit_counter[i] != 0)
		voxel_data[i] = mean_grid[i]/hit_counter[i];
}

void VNNKernel(const int nImages, const int imgWidth, const int imgHeight, 
		const float deltaGrid, const float deltaX, const float deltaY,
		const float xMin, const float yMin, const float zMin,
		const unsigned int gridIdx, const unsigned int gridIdy, const unsigned int gridIdz,
		const unsigned int voxelGridWidth, const unsigned int voxelGridHeight, const unsigned int voxelGridLength,
		float **offsets_d,
		float **rotations_d,
		unsigned char *char_image_data, 
		unsigned char *voxel_data, 
		unsigned short *mean_grid,
		unsigned char *hit_counter,
		cudaStream_t stream) {

	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(ceil(imgWidth/32.0f), ceil(imgHeight/32.0f), nImages);
	
	log_console.infoStream() << "[KERNEL::VNN] <<<" << toStringDim(dimBlock) << ", " << toStringDim(dimGrid) << ", " << 0 << ", " << stream << ">>>";

	VNN<<<dimGrid,dimBlock,0,stream>>>(nImages, imgWidth, imgHeight, 
			deltaGrid,  deltaX,  deltaY,
			xMin, yMin, zMin,
			gridIdx, gridIdy, gridIdz,
			voxelGridWidth,  voxelGridHeight,  voxelGridLength,
			offsets_d[0], offsets_d[1], offsets_d[2],
			rotations_d[0], rotations_d[1], rotations_d[2], 
			rotations_d[3], rotations_d[4], rotations_d[5],
			rotations_d[6], rotations_d[7], rotations_d[8],
			char_image_data, voxel_data, mean_grid, hit_counter);
	
	checkKernelExecution();
}
	
}
