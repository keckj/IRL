#include <cmath>
#include <cassert>
#include <iostream>
#include "utils/cudaUtils.hpp"

namespace kernel {

	__global__ void countVisibleQuads ( 
			unsigned int *d_counter,
			unsigned char *d_voxel_grid,
			const unsigned int gridWidth, const unsigned int gridHeight, const unsigned int gridLength,
			const unsigned char threshold) {

		*d_counter = 0;
		__syncthreads();

		unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
		unsigned int idz = blockIdx.z;
		unsigned int id = idz*gridWidth*gridHeight + idy*gridWidth + idx;


		if(idx >= gridWidth || idy >= gridHeight) {
			return;
		}

		__shared__ unsigned int localBlockCounter;
		localBlockCounter = 0;
		__syncthreads();

		unsigned char voxelValue = d_voxel_grid[id];	

		if(voxelValue > threshold) {
			unsigned char threadCounter = 0;
			threadCounter += (idx == 0 ? 1 : (d_voxel_grid[id - 1]<=threshold)); //left
			threadCounter += (idy == 0 ? 1 : (d_voxel_grid[id - gridWidth]<=threshold)); //down
			threadCounter += (idz == 0 ? 1 : (d_voxel_grid[id - gridHeight*gridWidth]<=threshold)); //back

			threadCounter += (idx == gridWidth  - 1 ? 1 : (d_voxel_grid[id + 1]<=threshold)); //right
			threadCounter += (idy == gridHeight - 1 ? 1 : (d_voxel_grid[id + gridWidth]<=threshold)); //up
			threadCounter += (idz == gridLength - 1 ? 1 : (d_voxel_grid[id + gridHeight*gridWidth]<=threshold)); //front

			atomicAdd(&localBlockCounter, threadCounter);
		}

		__syncthreads();
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			atomicAdd(d_counter, localBlockCounter);
		}

	}



	__device__ __inline__ void writeVec3f(float *d_array, unsigned int arrayID, unsigned char stride, float x, float y, float z) {
#pragma unroll
		for(int i = 0; i < stride; i++) {
			d_array[3*stride*arrayID + 3*i + 0] = x;
			d_array[3*stride*arrayID + 3*i + 1] = y;
			d_array[3*stride*arrayID + 3*i + 2] = z;
		}
	}

	__global__ void 
		__launch_bounds__(1024)
		computeVisibleQuads( 
				unsigned int *d_counter,
				float *d_quads, float *d_normals, float *d_colors,
				unsigned char *d_voxel_grid,
				const unsigned int gridWidth, const unsigned int gridHeight, const unsigned int gridLength,
				const float cube_w, const float cube_h, const float cube_d,
				const unsigned char threshold,
				const unsigned char normalStride, const unsigned char colorStride) {


			unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
			unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
			unsigned int idz = blockIdx.z % gridLength;
			unsigned int id = idz*gridWidth*gridHeight + idy*gridWidth + idx;


			if(idx >= gridWidth || idy >= gridHeight) {
				return;
			}

			unsigned char voxelValue = d_voxel_grid[id];	
			if (voxelValue <= threshold)
				return;

			unsigned char face = blockIdx.z/gridLength;
			bool draw = false;


			switch(face) {
				case(0): 
					{ //left
						draw = (idx == 0 ? true : (d_voxel_grid[id - 1] <= threshold)); //left
						break;
					}
				case(1): 
					{ //right
						draw =  (idx == gridWidth  - 1 ? true : (d_voxel_grid[id + 1] <= threshold)); //right
						break;
					}
				case(2): 
					{ //down
						draw = (idy == 0 ? true : (d_voxel_grid[id - gridWidth] <= threshold)); //down
						break;
					}
				case(3): 
					{ //up
						draw = (idy == gridHeight - 1 ? true : (d_voxel_grid[id + gridWidth] <= threshold)); //up
						break;
					}
				case(4): 
					{ //back
						draw = (idz == 0 ? true : (d_voxel_grid[id - gridHeight*gridWidth] <= threshold)); //back
						break;
					}
				case(5): 
					{ //front
						draw = (idz == gridLength - 1 ? true : (d_voxel_grid[id + gridHeight*gridWidth] <= threshold)); //front
						break;
					}
			}

			if(!draw)
				return;


			//get an array id
			const unsigned int arrayID = atomicAdd(d_counter, 1);

			//compute real position
			const float tx = idx*cube_w;
			const float ty = idy*cube_h;
			const float tz = idz*cube_d;

			/*write colors*/
			writeVec3f(d_colors, arrayID, colorStride, voxelValue/255.0f, voxelValue/255.0f, voxelValue/255.0f);

			switch(face) {
				case(0):
					{ //left
						writeVec3f(d_normals, arrayID, normalStride, -1.0f, 0.0f, 0.0f);
						writeVec3f(d_quads, 4*arrayID, 1, 0.0f + tx, 0.0f + ty, 0.0f + tz);
						writeVec3f(d_quads, 4*arrayID + 1, 1, 0.0f + tx, cube_h + ty, 0.0f + tz);
						writeVec3f(d_quads, 4*arrayID + 2, 1, 0.0f + tx, cube_h + ty, cube_d + tz);
						writeVec3f(d_quads, 4*arrayID + 3, 1, 0.0f + tx, 0.0f + ty, cube_d + tz);
						break;
					}
				case(1):
					{ //right
						writeVec3f(d_normals, arrayID, normalStride, +1.0f, 0.0f, 0.0f);
						writeVec3f(d_quads, 4*arrayID, 1, cube_w + tx, 0.0f + ty, 0.0f + tz);
						writeVec3f(d_quads, 4*arrayID + 1, 1, cube_w + tx, cube_h + ty, 0.0f + tz);
						writeVec3f(d_quads, 4*arrayID + 2, 1, cube_w + tx, cube_h + ty, cube_d + tz);
						writeVec3f(d_quads, 4*arrayID + 3, 1, cube_w + tx, 0.0f + ty, cube_d + tz);
						break;
					}
				case(2):
					{ //down
						writeVec3f(d_normals, arrayID, normalStride, 0.0f, -1.0f, 0.0f);
						writeVec3f(d_quads, 4*arrayID, 1, 0.0f + tx, 0.0f + ty, 0.0f + tz);
						writeVec3f(d_quads, 4*arrayID + 1, 1, 0.0f + tx, 0.0f + ty, cube_d + tz);
						writeVec3f(d_quads, 4*arrayID + 2, 1, cube_w + tx, 0.0f + ty, cube_d + tz);
						writeVec3f(d_quads, 4*arrayID + 3, 1, cube_w + tx, 0.0f + ty, 0.0f + tz);
						break;
					}
				case(3):
					{ //up
						writeVec3f(d_normals, arrayID, normalStride, 0.0f, 1.0f, 0.0f);
						writeVec3f(d_quads, 4*arrayID, 1, 0.0f + tx, cube_h + ty, 0.0f + tz);
						writeVec3f(d_quads, 4*arrayID + 1, 1, 0.0f + tx, cube_h + ty, cube_d + tz);
						writeVec3f(d_quads, 4*arrayID + 2, 1, cube_w + tx, cube_h + ty, cube_d + tz);
						writeVec3f(d_quads, 4*arrayID + 3, 1, cube_w + tx, cube_h + ty, 0.0f + tz);
						break;
					}
				case(4):
					{ //back
						writeVec3f(d_normals, arrayID, normalStride, 0.0f, 0.0f, -1.0f);
						writeVec3f(d_quads, 4*arrayID, 1, 0.0f + tx, 0.0f + ty, 0.0f + tz);
						writeVec3f(d_quads, 4*arrayID + 1, 1, cube_w + tx, 0.0f + ty, 0.0f + tz);
						writeVec3f(d_quads, 4*arrayID + 2, 1, cube_w + tx, cube_h + ty, 0.0f + tz);
						writeVec3f(d_quads, 4*arrayID + 3, 1, 0.0f + tx, cube_h + ty, 0.0f + tz);
						break;
					}
				case(5):
					{ //front
						writeVec3f(d_normals, arrayID, normalStride, 0.0f, 0.0f, 1.0f);
						writeVec3f(d_quads, 4*arrayID, 1, 0.0f + tx, 0.0f + ty, cube_d + tz);
						writeVec3f(d_quads, 4*arrayID + 1, 1, cube_w + tx, 0.0f + ty, cube_d + tz);
						writeVec3f(d_quads, 4*arrayID + 2, 1, cube_w + tx, cube_h + ty, cube_d + tz);
						writeVec3f(d_quads, 4*arrayID + 3, 1, 0.0f + tx, cube_h + ty, cube_d + tz);
						break;
					}
			}
		}

	
	void call_countVisibleQuads(dim3 dimGrid, dim3 dimBlock, unsigned int* nQuads_d, unsigned char *grid_d, unsigned int width, unsigned int height, unsigned int length, unsigned char threshold) {
			assert(nQuads_d);
			assert(grid_d);
			assert(width != 0);
			assert(height != 0);
			assert(length != 0);
			countVisibleQuads<<<dimGrid,dimBlock>>>(nQuads_d, grid_d, width, height, length, threshold);
			checkKernelExecution();
	}

}
