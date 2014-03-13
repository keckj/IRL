#include <cmath>
#include <cassert>
#include <iostream>
#include "kernels.hpp"
#include "utils/cudaUtils.hpp"

namespace kernel {

	__global__ void countVisibleQuads ( 
			unsigned int *d_counter,
			unsigned char *d_voxel_grid,
			const unsigned int gridWidth, const unsigned int gridHeight, const unsigned int gridLength,
			const unsigned char threshold) {
	
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

	unsigned int computeQuads(
			float **h_quads, float **h_normals, float **h_colors, 
			unsigned char* d_voxel_grid,
			const unsigned int gridWidth, const unsigned int gridHeight, const unsigned int gridLength,
			const float cube_w, const float cube_h, const float cube_d,
			const unsigned char threshold, 
			const NormalType nt, const ColorType ct) {

		dim3 dimBlock(32, 32, 1);
		dim3 dimGrid(ceil(gridWidth/32.0f), ceil(gridHeight/32.0f), gridLength);

		//compute number of quads
		unsigned int *h_nQuads = new unsigned int[1];
		unsigned int *d_nQuads;

		CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_nQuads, sizeof(unsigned int)));
		CHECK_CUDA_ERRORS(cudaMemset(d_nQuads, 0, sizeof(unsigned int)));

		countVisibleQuads<<<dimGrid,dimBlock>>>(d_nQuads, d_voxel_grid, gridWidth, gridHeight, gridLength, threshold);
		checkKernelExecution();
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
		CHECK_CUDA_ERRORS(cudaMemcpy(h_nQuads, d_nQuads, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		
		/*log_console.infoStream() << "Nquads avant génération " << *h_nQuads;*/

		if(*h_nQuads == 0) {
			*h_quads = 0;
			*h_colors = 0;
			*h_normals = 0;
			return 0;
		}

		//compute quads
		float *d_quads, *d_normals, *d_colors;
		const unsigned int quadSize = (*h_nQuads)*4*3*sizeof(float);

		const unsigned char normalStride = (nt == NORMAL_PER_QUAD ? 1 : 4);
		const unsigned char colorStride = (ct == COLOR_PER_QUAD ? 1 : 4);
		const unsigned int normalSize = normalStride*(*h_nQuads)*3*sizeof(float);
		const unsigned int colorSize = colorStride*(*h_nQuads)*3*sizeof(float);

		//quads
		CHECK_CUDA_ERRORS(cudaMallocHost((void**) h_quads, quadSize));
		CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_quads, quadSize));

		//normals
		CHECK_CUDA_ERRORS(cudaMallocHost((void**) h_normals, normalSize));
		CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_normals, normalSize));

		//colors
		CHECK_CUDA_ERRORS(cudaMallocHost((void**) h_colors, colorSize));
		CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_colors, colorSize));

		//counter
		unsigned int *h_counter = new unsigned int[1];
		CHECK_CUDA_ERRORS(cudaMemset(d_nQuads, 0, sizeof(unsigned int)));

		//compute 

		dimBlock = dim3(32,32,1);
		dimGrid = dim3(ceil(gridWidth/32.0f), ceil(gridHeight/32.0f), 6*gridLength);
		computeVisibleQuads<<<dimGrid,dimBlock>>>(
				d_nQuads, 
				d_quads, d_normals, d_colors,
				d_voxel_grid, 
				gridWidth, gridHeight, gridLength,
				cube_w, cube_h, cube_d,
				threshold,
				normalStride, colorStride);

		checkKernelExecution();
					
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

		//verify 
		CHECK_CUDA_ERRORS(cudaMemcpy(h_counter, d_nQuads, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		/*log_console.infoStream() << "Nquads après génération " << *h_counter;*/
		

		assert((*h_counter) == (*h_nQuads));

		//copy data back to CPU
		CHECK_CUDA_ERRORS(cudaMemcpy(*h_quads, d_quads, quadSize, cudaMemcpyDeviceToHost));
		CHECK_CUDA_ERRORS(cudaMemcpy(*h_normals, d_normals, normalSize, cudaMemcpyDeviceToHost));
		CHECK_CUDA_ERRORS(cudaMemcpy(*h_colors, d_colors, colorSize, cudaMemcpyDeviceToHost));

		//keep data before free
		const unsigned int nQuads = *(h_nQuads);

		//free remaining data
		CHECK_CUDA_ERRORS(cudaFree(d_nQuads));
		CHECK_CUDA_ERRORS(cudaFree(d_quads));
		CHECK_CUDA_ERRORS(cudaFree(d_normals));
		CHECK_CUDA_ERRORS(cudaFree(d_colors));

		free(h_counter);
		free(h_nQuads);

		return nQuads;
	}

}
