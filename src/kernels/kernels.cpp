
#include <stdio.h>
#include <cassert>

#include "kernels.hpp"
#include "utils/cudaUtils.hpp"

#include "memoryManager/PinnedCPUResource.hpp"
#include "memoryManager/PagedCPUResource.hpp"
#include "memoryManager/GPUResource.hpp"
#include "memoryManager/GPUMemory.hpp"
#include "memoryManager/CPUMemory.hpp"
#include "memoryManager/sharedResource.hpp"

namespace kernel {

	extern void castKernel(unsigned long dataSize, float *float_data, unsigned char *char_data);

	extern void call_countVisibleQuads(dim3 dimGrid, dim3 dimBlock, 
			unsigned int* nQuads_d, unsigned char *grid_d, 
			unsigned int width, unsigned int height, unsigned int length, 
			unsigned char threshold);

	extern void call_computeVisibleQuads(dim3 dimGrid, dim3 dimBlock,
			unsigned int *d_counter,
			float *d_quads, float *d_colors,
			unsigned char *d_voxel_grid,
			const unsigned int gridIdx, const unsigned int gridIdy, const unsigned int gridIdz,
			const unsigned int gridWidth, const unsigned int gridHeight, const unsigned int gridLength,
			const float voxelWidth,
			const unsigned char threshold,
			const unsigned char colorStride);


	unsigned int computeQuads(float **th_quads, float **th_colors, 
			VoxelGridTree<unsigned char, PinnedCPUResource, GPUResource> *cpuGrid,
			float alpha, 
			const unsigned char threshold, 
			const NormalType nt, const ColorType ct) {

		
		cudaSetDevice(0);
		dim3 dimBlock(32, 32, 1);
		dim3 dimGrid(ceil(cpuGrid->subwidth()/32.0f), ceil(cpuGrid->subheight()/32.0f), cpuGrid->sublength());

		//compute number of quads
		SharedResource<unsigned int, PinnedCPUResource, GPUResource> nQuads_s(0, 1ul);	
		nQuads_s.allocateAll();

		GPUResource<unsigned char> subgrid_d(0, cpuGrid->subgridSize());
		subgrid_d.allocate();

		unsigned int *counts = new unsigned int[cpuGrid->nChilds()];
		unsigned int totalQuads = 0;

		int i = 0;
		for(auto it = cpuGrid->begin(); it != cpuGrid->end(); ++it)
		{
			PinnedCPUResource<unsigned char> subgrid_h(**it);
			SharedResource<unsigned char, PinnedCPUResource, GPUResource> subgrid_s(subgrid_h, subgrid_d);	
			subgrid_s.copyToDevice();

			CHECK_CUDA_ERRORS(cudaMemset(nQuads_s.deviceData(), 0, nQuads_s.dataBytes()));

			call_countVisibleQuads(dimGrid,dimBlock,nQuads_s.deviceData(), subgrid_s.deviceData(), cpuGrid->subwidth(), cpuGrid->subheight(), cpuGrid->sublength(), threshold);

			checkKernelExecution();

			nQuads_s.copyToHost();
			cudaDeviceSynchronize();	


			counts[i] = *(nQuads_s.hostData());
			totalQuads += counts[i];
			i++;
		}

		log_console.debugStream() << "Nquads avant génération " << totalQuads;

		if(totalQuads == 0) {
			*th_quads = 0;
			*th_colors = 0;
			return 0;
		}


		//compute quads

		const unsigned char colorStride = (ct == COLOR_PER_QUAD ? 1 : 4);

		const unsigned int quadSize = totalQuads*4*3*sizeof(float);
		const unsigned int colorSize = colorStride*totalQuads*3*sizeof(float);

		//CPU quads colors
		CHECK_CUDA_ERRORS(cudaMallocHost((void**) th_quads, quadSize));
		CHECK_CUDA_ERRORS(cudaMallocHost((void**) th_colors, colorSize));

		//pointers
		float *p_quads = *th_quads;
		float *p_colors = *th_colors;

		//counters
		unsigned int totalQuads2 = 0;
		unsigned int *counts2 = new unsigned int[cpuGrid->nChilds()];

		//compute 
		dimBlock = dim3(32,32,1);
		dimGrid = dim3(ceil(cpuGrid->subwidth()/32.0f), ceil(cpuGrid->subheight()/32.0f), 6*cpuGrid->sublength());

		unsigned int nGrid = 0;
		unsigned int gridIdx = 0, gridIdy = 0,  gridIdz = 0;
		for (auto it = cpuGrid->begin(); it != cpuGrid->end(); ++it) {

			if(counts[nGrid] != 0) {

				PinnedCPUResource<unsigned char> subgrid_h(**it);
				SharedResource<unsigned char, PinnedCPUResource, GPUResource> subgrid_s(subgrid_h, subgrid_d);	
				subgrid_s.copyToDevice();

				CHECK_CUDA_ERRORS(cudaMemset(nQuads_s.deviceData(), 0, nQuads_s.dataBytes()));

				PinnedCPUResource<float> h_quads(p_quads, counts[nGrid]*4*3);
				PinnedCPUResource<float> h_colors(p_colors, counts[nGrid]*colorStride*3);
				SharedResource<float, PinnedCPUResource, GPUResource> s_quads(h_quads, 0);
				SharedResource<float, PinnedCPUResource, GPUResource> s_colors(h_colors, 0);
				s_quads.allocateOnDevice();
				s_colors.allocateOnDevice();

				call_computeVisibleQuads(
						dimGrid, dimBlock,
						nQuads_s.deviceData(), 
						s_quads.deviceData(), s_colors.deviceData(),
						subgrid_s.deviceData(),
						gridIdx, gridIdy, gridIdz, 
						cpuGrid->subwidth(), cpuGrid->subheight(), cpuGrid->sublength(),
						alpha*cpuGrid->voxelSize(), 
						threshold,
						colorStride);

				nQuads_s.copyToHost();
				CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

				counts2[nGrid] = *(nQuads_s.hostData());
				totalQuads2 += counts2[nGrid];


				//verify the number of generated quads
				assert(counts[nGrid] == counts2[nGrid]);

				//copy back data and actualize pointers
				s_quads.copyToHost();
				s_colors.copyToHost();
				CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

				p_quads += s_quads.dataSize();
				p_colors += s_colors.dataSize();

			}

			//switch to next subgrid
			nGrid++;
			gridIdz = nGrid / (cpuGrid->nGridX() * cpuGrid->nGridY());
			gridIdy = (nGrid % (cpuGrid->nGridX() * cpuGrid->nGridY())) / cpuGrid->nGridX();
			gridIdx = nGrid % cpuGrid->nGridX();
		}

		log_console.debugStream() << "Nquads après génération " << totalQuads2;
		assert(totalQuads == totalQuads2);

		delete [] counts;
		delete [] counts2;

		return totalQuads;
	}

}
