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


	__device__ unsigned short _atomicAddShort(unsigned short* address, unsigned short val) {
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

	__device__ unsigned char _atomicAddChar(unsigned char* address, unsigned char val) {

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
	computeMean(unsigned char *grid, unsigned int *hit_counter, unsigned int *sum, unsigned long long int nData, unsigned long long int offset) {
			unsigned int localId = 65535u*512u*blockIdx.y + 512u*blockIdx.x + threadIdx.x;  
			unsigned long long int id = offset + localId;
				
			if(id >= nData) {
				return;
			}
			
			if(hit_counter[id] == 0u)
				grid[id] = 0u;
			else
				grid[id] = sum[id]/hit_counter[id];
		}
	
	void computeMeanKernel(unsigned char *grid, unsigned int *hit_counter, unsigned int *sum, 
			const unsigned long nData, cudaStream_t stream) {
		
		/*Découpe en plusieurs kernels car on ne peux 'que' 
		  lancer 65535x65535 threads à la fois en compute capability 1.x*/
		unsigned long maxThreads = 512ul*65535ul;
		dim3 dimBlock(512);
		dim3 dimGrid(ceil(nData/512.0), ceil(nData/(512.0*65535.0)));
		
		unsigned int nPass = (nData+maxThreads-1ul)/maxThreads;

		for(unsigned int i=0; i < nPass; i++) {
			computeMean<<<dimGrid,dimBlock,0,stream>>>(grid, hit_counter, sum, nData, i*maxThreads);
		}
	}

	__global__ void
	__launch_bounds__(512)
	HoleFilling(const unsigned char interpolationRadius, 
			const unsigned int gridIdx, const unsigned int gridIdy, const unsigned int gridIdz,
			const unsigned int voxelGridWidth, const unsigned int voxelGridHeight, const unsigned int voxelGridLength,
			const unsigned long long int nData,
			const unsigned long long int offset,
			unsigned char *srcGrid, unsigned char *dstGrid)
	{
		
		unsigned int iid = 65535u*512u*blockIdx.y + 512u*blockIdx.x + threadIdx.x;  
		unsigned long long int id = offset + iid;

		if(id >= nData)
			return;
	
		//cas ou il n'y a pas de trou
		if(srcGrid[id] != 0)
			dstGrid[id] = srcGrid[id];

		unsigned long long int idz = id / (voxelGridHeight*voxelGridWidth);
		unsigned long long int idy = (id % (voxelGridHeight*voxelGridWidth))/voxelGridWidth;
		unsigned long long int idx = id % voxelGridWidth;
		

		//gestion des bords effectué sur CPU
		if(        idx < interpolationRadius - 1u 
			|| idy < interpolationRadius - 1u
			|| idz < interpolationRadius - 1u
			|| idx > (voxelGridWidth - interpolationRadius)
			|| idy > (voxelGridWidth - interpolationRadius)
			|| idz > (voxelGridWidth - interpolationRadius)) {
			return;
		}


		unsigned char localValue;
		unsigned short localHit;
		unsigned int localSum;

		unsigned long long int localId;
		for(int i = -interpolationRadius+1; i <= interpolationRadius-1; i++) {
			for(int j = -interpolationRadius+1; j <= interpolationRadius-1; j++) {
				for(int k = -interpolationRadius+1; k<= interpolationRadius-1; k++) {
					localId = id + i*voxelGridHeight*voxelGridWidth + j*voxelGridWidth + k;
					localValue = srcGrid[localId]; 

					if(localValue!=0) {
						localHit++;
						localSum += localValue;
					}
				}
			}
		}

		if(localHit != 0) 
			dstGrid[id] = localSum/localHit;	
	}
		

	__global__ void 
	__launch_bounds__(512)
		VNN(const int nImages, const int imgWidth, const int imgHeight, 
				const float deltaGrid, const float deltaX, const float deltaY,
				const float xMin, const float yMin, const float zMin,
				const unsigned int gridIdx, const unsigned int gridIdy, const unsigned int gridIdz,
				const unsigned int voxelGridWidth, const unsigned int voxelGridHeight, const unsigned int voxelGridLength,
				float *offsetX, float *offsetY, float *offsetZ,
				float *r1, float *r2, float *r3, float *r4, float *r5, float *r6, float *r7, float *r8, float *r9,
				unsigned char *char_image_data, 
				unsigned char *voxel_data, 
				unsigned int *mean_grid,
				unsigned int *hit_counter) {

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
			if(! 
				(ix/voxelGridWidth == gridIdx 
				&& iy/voxelGridHeight == gridIdy
				&& iz/voxelGridLength == gridIdz)) {
				return;
			}

			ix %= voxelGridWidth;
			iy %= voxelGridHeight;
			iz %= voxelGridLength;
			unsigned long long int i = iz*voxelGridHeight*voxelGridWidth + iy*voxelGridWidth + ix;

			unsigned char value = char_image_data[id];
		
			//Ancien code pour les hit_grid unsigned char ou unsigned short
			//à cause du risque d'overflow si trop de pixel dans le même voxel
			/*unsigned short current, old;*/
			/*do {*/
				/*old = hit_counter[i];*/
				/*if(old == 65536)*/
					/*return;*/
				/*current = _atomicCASShort(hit_counter + i, old, old+1u);*/
			/*} while(current != old);*/
			
			atomicAdd(hit_counter + i, 1);
			atomicAdd(mean_grid + i, (unsigned int)value);
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
			unsigned int *mean_grid,
			unsigned int *hit_counter,
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

