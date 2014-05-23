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
	__launch_bounds__(1024)
	computeMean(unsigned char *grid, unsigned int *hit_counter, unsigned int *sum, unsigned long long int nData, unsigned long long int offset) {
			unsigned long long int localId = (32ull*blockIdx.y + threadIdx.y) * 65504ull 
							 + 32ull*blockIdx.x + threadIdx.x;  

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
			const unsigned long long int nData, cudaStream_t stream) {
		
		/*Découpe en plusieurs kernels car on ne peux 'que' 
		  lancer 65535x65535 threads à la fois en compute capability 1.x*/
		unsigned long long int maxThreads = 65504ull*65504ull*32ull*32ull;

		dim3 dimBlock(32,32);
		dim3 dimGrid(min((nData+32ull-1ull         )/(32ull         ),65504ull),
			     min((nData+32ull*65504ull-1ull)/(32ull*65504ull),65504ull));
		
		unsigned int nPass = (nData+maxThreads-1ull)/maxThreads;
		
		for(unsigned int i=0; i < nPass-1; i++) {
			computeMean<<<dimGrid,dimBlock,0,stream>>>(grid, hit_counter, sum, nData, i*maxThreads);
		}
		computeMean<<<dimGrid,dimBlock,0,stream>>>(grid, hit_counter, sum, nData, (nPass-1)*maxThreads);
	}
	
	
	__global__ void
	__launch_bounds__(1024)
	HoleFilling(const unsigned char interpolationRadius, 
			const unsigned int gridIdx, const unsigned int gridIdy, const unsigned int gridIdz,
			const unsigned int subgridWidth, const unsigned int subgridHeight, const unsigned int subgridLength,
			const unsigned long long int nData, const unsigned long long int offset,
			unsigned char *srcGrid, unsigned char *dstGrid)
	{
		

		unsigned long long int threadId = offset 
						+ (32ull*blockIdx.y + threadIdx.y) * 65504ull 
						+ 32ull*blockIdx.x + threadIdx.x;  
		unsigned long long int threadIdZ = threadId / (subgridHeight*subgridWidth);
		unsigned long long int threadIdY = (threadId % (subgridHeight*subgridWidth))/subgridWidth;
		unsigned long long int threadIdX = threadId % subgridWidth;
		
		unsigned long long int blockId = offset + (65504ull*blockIdx.y + blockIdx.x)*32ull;
		unsigned long long int blockIdZ = blockId / (subgridHeight*subgridWidth);
		unsigned long long int blockIdY = (blockId % (subgridHeight*subgridWidth))/subgridWidth;
		unsigned long long int blockIdX = blockId % subgridWidth;
		
		extern __shared__ unsigned char locValues[];
		unsigned int locGridWidth, locGridHeight, locGridLength;
		locGridWidth = 32ul+2ul*interpolationRadius;
		locGridHeight = 32ul+2ul*interpolationRadius;
		locGridLength = 1ul+2ul*interpolationRadius;

		unsigned int nNeighbors = locGridWidth * locGridHeight * locGridLength;

		unsigned int locGridIdx, locGridIdy, locGridIdz, locGridId;
		locGridIdx = interpolationRadius + threadIdx.x;
		locGridIdy = interpolationRadius + threadIdx.y;
		locGridIdz = interpolationRadius;
		locGridId = locGridIdz*locGridWidth*locGridHeight + locGridIdy*locGridWidth + locGridIdx;
	
		for(int i = 0; i < (nNeighbors+1023)/1024; i++) {
			unsigned int locPos = 1024*i + 32*threadIdx.y + threadIdx.x;
			unsigned int locPosIdz = locPos/(locGridHeight*locGridWidth);
			unsigned int locPosIdy = (locPos%(locGridHeight*locGridWidth))/locGridWidth;
			unsigned int locPosIdx = locPos%locGridWidth;

			unsigned long long int globalPos = (blockIdZ-interpolationRadius+locPosIdz)*subgridWidth*subgridHeight
							+ (blockIdY-interpolationRadius+locPosIdy)*subgridWidth
							+ blockIdX-interpolationRadius+locPosIdx;

			if(locPos < nNeighbors && globalPos < nData)
				locValues[locPos] = srcGrid[globalPos];
		}

		__syncthreads();
		
		if(threadId >= nData)
			return;
		
		//cas ou il n'y a pas de trou
		if(srcGrid[threadId] != 0u) {
			dstGrid[threadId] = srcGrid[threadId];
			return;
		}
		
		//gestion des bords effectué sur CPU
		if(        threadIdX < interpolationRadius 
			|| threadIdY < interpolationRadius
			|| threadIdZ < interpolationRadius
			|| threadIdX >= (subgridWidth - interpolationRadius)
			|| threadIdY >= (subgridHeight - interpolationRadius)
			|| threadIdZ >= (subgridLength - interpolationRadius)) {
			dstGrid[threadId]=0;
			return;
		}

		
		/*unsigned int localValue;*/
		float localHit = 0.0f;
		float localSum = 0.0f;
		unsigned char localValue;
		unsigned long long int localId;

		unsigned int r3 = interpolationRadius*interpolationRadius*interpolationRadius;
		unsigned int d3;
		float invd, invd3;

		for(int i = -interpolationRadius; i <= interpolationRadius; i++) {
			for(int j = -interpolationRadius; j <= interpolationRadius; j++) {
				for(int k = -interpolationRadius; k<= interpolationRadius; k++) {
					d3 = i*i+j*j+k*k;	
					invd = __frsqrt_rn(__uint2float_rn(d3));
					invd3 = invd*invd*invd;
					
					if(d3 != 0 && d3 <= r3) {
						/*localId = id + i*(int)subgridHeight*(int)subgridWidth + j*(int)subgridWidth + k;*/
						/*localValue = srcGrid[localId];*/
						localId = locGridId + i*(int)locGridHeight*(int)locGridWidth + j*(int)locGridWidth+k;
						localValue = locValues[localId];

						if(localValue > 0) {
							localHit += invd3;
							localSum += invd3*localValue;
						}
					}
				}
			}
		}

		if(localHit > 0.0f)
			dstGrid[threadId] = __float2uint_rn(localSum/localHit);	
		else
			dstGrid[threadId] = 0;
	}
		
	void HoleFillingKernel(const unsigned char interpolationRadius,
			const unsigned int gridIdx, const unsigned int gridIdy, const unsigned int gridIdz,
			const unsigned int subgridWidth, const unsigned int subgridHeight, const unsigned int subgridLength,
			const unsigned long long int nData,
			unsigned char *srcGrid, 
			unsigned char *dstGrid, 
			cudaStream_t stream) {

		unsigned long long int maxThreads = 65504ull*65504ull*32ull*32ull;

		dim3 dimBlock(32,32);
		dim3 dimGrid(min((nData+32ull-1ull         )/(32ull         ),65504ull),
			     min((nData+32ull*65504ull-1ull)/(32ull*65504ull),65504ull));
		
		unsigned int nPass = (nData+maxThreads-1ull)/maxThreads;
		
		unsigned int nNeighbors = (32ul+2*interpolationRadius)*(32ul+2*interpolationRadius)*(1ul+2*interpolationRadius);
		unsigned int sharedMemoryBytes = nNeighbors*sizeof(unsigned char);

		
		log_console.infoStream() << "[KERNEL::Hole Filling] <<<" << toStringDim(dimBlock) << ", " << toStringDim(dimGrid) << ", " << toStringMemory(sharedMemoryBytes) << ", " << stream << ">>>";
		
		for(unsigned int i=0; i < nPass-1; i++) {
			HoleFilling<<<dimGrid,dimBlock,sharedMemoryBytes,stream>>>(interpolationRadius,
					gridIdx, gridIdy, gridIdz,
					subgridWidth,  subgridHeight,  subgridLength,
					maxThreads, i*maxThreads,
					srcGrid, dstGrid);
		}
			
		HoleFilling<<<dimGrid,dimBlock,sharedMemoryBytes,stream>>>(interpolationRadius,
					gridIdx, gridIdy, gridIdz,
					subgridWidth,  subgridHeight,  subgridLength,
					nData%maxThreads, (nPass-1)*maxThreads,
					srcGrid, dstGrid);

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

	}

}

