#include <cmath>

__global__ void test(const int nImages, const int imgWidth, const int imgHeight, float *float_data, unsigned char *char_data) {
	
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;    // this thread handles the data at its thread id
	unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;    // this thread handles the data at its thread id
	unsigned int id = idy*imgWidth + idx;

	if(idx >= imgWidth || idy >= nImages*imgHeight)
		return;
	
	char_data[id] = (unsigned char) float_data[id];
}

void testKernel(const int nImages, const int imgWidth, const int imgHeight, float *float_data, unsigned char *char_data) {
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(ceil(imgWidth/32), ceil(nImages*imgHeight/32), 1);

	test<<<dimGrid,dimBlock>>>(nImages, imgWidth, imgHeight, float_data, char_data);
}
