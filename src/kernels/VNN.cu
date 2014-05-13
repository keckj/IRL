#include <cmath>

#include "utils/cudaUtils.hpp"
#include "utils/log.hpp"
#include "utils/utils.hpp"

namespace kernel {

	__global__ void 
	__launch_bounds__(512)
	cast(const long dataSize, float *float_data, unsigned char *char_data) {
		unsigned int id = blockIdx.y*65535*512 
			+ blockIdx.x*512 + threadIdx.x;

		if(id > dataSize)
			return;

		char_data[id] = (unsigned char) float_data[id];
	}

	void castKernel(unsigned long dataSize, float *float_data, unsigned char *char_data) {
		dim3 dimBlock(512);
		dim3 dimGrid((unsigned int)ceil(dataSize/512.0) % 65535, ceil(dataSize/(512*65535.0f)));
		log_console.infoStream() << "[KERNEL::Cast] <<<" << toStringDim(dimBlock) << ", " << toStringDim(dimGrid) << ">>>";
		cast<<<dimGrid,dimBlock>>>(dataSize, float_data, char_data);
	}


	__device__ char _atomicAddShort(unsigned short* address, unsigned short val) {
		unsigned int *base_address = (unsigned int *)((size_t)address & ~2);
		unsigned int old, assumed, sum, new_;

		old = *base_address;
		do {
			assumed = old;
			sum = val + (unsigned short)__byte_perm(old, 0, ((size_t)address & 2) ? 0x4432 : 0x4410);
			new_ = __byte_perm(old, sum, ((size_t)address & 2) ? 0x5410 : 0x3254);
			old = atomicCAS(base_address, assumed, new_);
		} while (assumed != old);

		return old;
	}

	__device__ char _atomicAddChar(unsigned char* address, unsigned char val) {

		unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
		unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
		unsigned int sel = selectors[(size_t)address & 3];
		unsigned int old, assumed, sum, new_;

		old = *base_address;
		do {
			assumed = old;
			sum = val + (unsigned short)__byte_perm(old, 0, ((size_t)address & 3));
			new_ = __byte_perm(old, sum, sel);
			old = atomicCAS(base_address, assumed, new_);
		} while (assumed != old);

		return old;
	}

	__global__ void 
	__launch_bounds__(512)
	computeMean(unsigned char *grid, unsigned char *hit_counter, unsigned short *sum, unsigned long nData) {
			unsigned int id = 512*65535*blockIdx.y + 512*blockIdx.x + threadIdx.x;  

			if(id >= nData)
				return;
			
			if(hit_counter[id] == 0)
				return;

			grid[id] = sum[id]/hit_counter[id];
		}


	__global__ void 
	__launch_bounds__(1024)
		VNN(const int nImages, const int imgWidth, const int imgHeight, 
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

			//check if the pixel is in the subgrid
			if(! (ix/voxelGridWidth == gridIdx 
						&& iy/voxelGridHeight == gridIdy
						&& iz/voxelGridLength == gridIdz)) {
				return;
			}

			ix %= voxelGridWidth;
			iy %= voxelGridHeight;
			iz %= voxelGridLength;
			unsigned long i = iz*voxelGridHeight*voxelGridWidth + iy*voxelGridWidth + ix;

			if(hit_counter[i] > 255)
				return;

			unsigned char value = char_image_data[id];

			_atomicAddChar(hit_counter + i, 1);	
			_atomicAddShort(mean_grid + i, (unsigned short)value);
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

	void computeMeanKernel(unsigned char *grid, unsigned char *hit_counter, unsigned short *sum, 
			const unsigned long nData, cudaStream_t stream) {
		dim3 dimBlock(512);
		dim3 dimGrid(((unsigned int)ceil(nData/512.0))%65535, ceil(nData/(512*65535.0f)));

		computeMean<<<dimGrid,dimBlock,0,stream>>>(grid, hit_counter, sum, nData);
	}
}

