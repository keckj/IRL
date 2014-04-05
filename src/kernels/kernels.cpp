
#include <stdio.h>
#include "kernels.hpp"

#include "memoryManager/PinnedCPUResource.hpp"
#include "memoryManager/PagedCPUResource.hpp"
#include "memoryManager/GPUResource.hpp"
#include "memoryManager/GPUMemory.hpp"
#include "memoryManager/CPUMemory.hpp"
#include "memoryManager/sharedResource.hpp"
			
namespace kernel {
	
	
	extern void call_countVisibleQuads(dim3 dimGrid, dim3 dimBlock, unsigned int* nQuads_d, unsigned char *grid_d, unsigned int width, unsigned int height, unsigned int length, unsigned char threshold);


	unsigned int computeQuads(float **h_quads, float **h_normals, float **h_colors, 
			VoxelGridTree<unsigned char, PinnedCPUResource, GPUResource> *cpuGrid,
			unsigned char threshold, 
			NormalType nt, ColorType ct) {

		cudaSetDevice(0);

		dim3 dimBlock(32, 32, 1);
		dim3 dimGrid(ceil(cpuGrid->subwidth()/32.0f), ceil(cpuGrid->subheight()/32.0f), cpuGrid->sublength());

		//compute number of quads
		SharedResource<unsigned int, PinnedCPUResource, GPUResource> nQuads_s(0, 1ul);	
		nQuads_s.allocateAll();

		GPUResource<unsigned char> subgrid_d(0, cpuGrid->subgridSize());
		subgrid_d.allocate();

		unsigned int *counts = new unsigned int[cpuGrid->nChilds()];

		int i = 0;
		for(auto it = cpuGrid->begin(); it != cpuGrid->end(); ++it)
		{
			PinnedCPUResource<unsigned char> subgrid_h(**it);
			SharedResource<unsigned char, PinnedCPUResource, GPUResource> subgrid_s(subgrid_h, subgrid_d);	
			subgrid_s.copyToDevice();

			call_countVisibleQuads(dimGrid,dimBlock,nQuads_s.deviceData(), subgrid_s.deviceData(), cpuGrid->subwidth(), cpuGrid->subheight(), cpuGrid->sublength(), threshold);

			nQuads_s.copyToHost();
			counts[i] = *(nQuads_s.hostData());
			log_console.infoStream() << "Nquads avant génération pour la grille " << i << " =>\t" << *(nQuads_s.hostData());
			i++;
		}


		if(nQuads_s.hostData() == 0) {
			*h_quads = 0;
			*h_colors = 0;
			*h_normals = 0;
			return 0;
		}

		/*
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
		dimGrid = dim3(ceil(cpuGrid->subwidth()/32.0f), ceil(cpuGrid->subheight()/32.0f), 6*cpuGrid->sublength());
		computeVisibleQuads<<<dimGrid,dimBlock>>>(
		d_nQuads, 
		d_quads, d_normals, d_colors,
		d_voxel_grid, 
		cpuGrid->subwidth(), cpuGrid->subheight(), cpuGrid->sublength(),
		cube_w, cube_h, cube_d,
		threshold,
		normalStride, colorStride);

		checkKernelExecution();

		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

		//verify 
		CHECK_CUDA_ERRORS(cudaMemcpy(h_counter, d_nQuads, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		log_console.infoStream() << "Nquads après génération " << *h_counter;


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

*/

		const unsigned int nQuads = *(nQuads_s.hostData());
		return nQuads;
	}

}
