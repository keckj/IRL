#include <cmath>
#include "kernels.hpp"
#include "utils/cudaUtils.hpp"

namespace kernel {

__global__ void test(const int nImages, const int imgWidth, const int imgHeight, float *float_data, unsigned char *char_data) {
	
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;    // this thread handles the data at its thread id
	unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;    // this thread handles the data at its thread id
	unsigned int id = blockIdx.z*imgWidth*imgHeight + idy*imgWidth + idx;

	if(idx >= imgWidth || idy >= imgHeight)
		return;
	
	char_data[id] = (unsigned char) float_data[id];
}

void testKernel(const int nImages, const int imgWidth, const int imgHeight, float *float_data, unsigned char *char_data) {
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(ceil(imgWidth/32.0f), ceil(imgHeight/32.0f), nImages);

	test<<<dimGrid,dimBlock>>>(nImages, imgWidth, imgHeight, float_data, char_data);
}



__global__ void VNN(const int nImages, const int imgWidth, const int imgHeight, 
		const float deltaGrid, const float deltaX, const float deltaY,
		const float xMin, const float yMin, const float zMin,
		const unsigned int voxelGridWidth, const unsigned int voxelGridHeight, const unsigned int voxelGridLength,
		float *offsetX, float *offsetY, float *offsetZ,
		float *r1, float *r2, float *r3, float *r4, float *r5, float *r6, float *r7, float *r8, float *r9,
		unsigned char *char_image_data, unsigned char *voxel_data, unsigned char *hit_counter) {
	
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;    // this thread handles the data at its thread id
	unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;    // this thread handles the data at its thread id
	unsigned int id = idy*imgWidth + idx;
	
	unsigned int n = blockIdx.z;

	if(idx >= imgWidth || idy >= imgHeight)
		return;

	float vx = idx*deltaX;	
	float vy = idy*deltaY;
	
	float px = r1[n]*vx + r2[n]*vy + r3[n]*0.0f + offsetX[n] - xMin;
	float py = r4[n]*vx + r5[n]*vy + r6[n]*0.0f + offsetY[n] - yMin;
	float pz = r7[n]*vx + r8[n]*vy + r9[n]*0.0f + offsetZ[n] - zMin;

	unsigned int ix = __float2uint_rd(px/deltaGrid);
	unsigned int iy = __float2uint_rd(py/deltaGrid);
	unsigned int iz = __float2uint_rd(pz/deltaGrid);
	
	unsigned long i = iz*voxelGridHeight*voxelGridWidth + iy*voxelGridWidth + ix;

	unsigned char value = char_image_data[id];
	unsigned char hit = hit_counter[i];
	
#ifdef _VOXEL_MEAN_VALUE
	float mean;
	if(hit == 255) {
		return;
	}
	else if(hit == 0) {
		voxel_data[i] = value;
		hit_counter[i] = hit + 1;
	}
	else {
		mean = ((int)hit*(int)voxel_data[i] + value)/(hit + 1);
		voxel_data[i] = (unsigned char) mean;
		hit_counter[i] = hit + 1;
	}
#else
	if (hit != 0 && value > voxel_data[i]) {
		voxel_data[i] = value;
	}
	else {
		hit_counter[i] = 1;
		voxel_data[i] = value;
	}
#endif

}

void VNNKernel(const int nImages, const int imgWidth, const int imgHeight, 
		const float deltaGrid, const float deltaX, const float deltaY,
		const float xMin, const float yMin, const float zMin,
		const unsigned int voxelGridWidth, const unsigned int voxelGridHeight, const unsigned int voxelGridLength,
		float *offsetX, float *offsetY, float *offsetZ,
		float *r1, float *r2, float *r3, float *r4, float *r5, float *r6, float *r7, float *r8, float *r9,
		unsigned char *char_image_data, unsigned char *voxel_data, unsigned char *hit_counter) {

	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(ceil(imgWidth/32.0f), ceil(imgHeight/32.0f - 1), nImages);

	VNN<<<dimGrid,dimBlock>>>(nImages, imgWidth, imgHeight, 
			deltaGrid,  deltaX,  deltaY,
			xMin, yMin, zMin,
			voxelGridWidth,  voxelGridHeight,  voxelGridLength,
			offsetX, offsetY, offsetZ,
			r1, r2, r3, r4, r5, r6, r7, r8, r9,
			char_image_data, voxel_data, hit_counter);
}
	
}
