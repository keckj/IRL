
#ifndef _KERNELS_H_
#define _KERNELS_H_

namespace kernel {

void testKernel(const int nImages, const int imgWidth, const int imgHeight, float *float_data, unsigned char *char_data);

void VNNKernel(
		const int nImages, const int imgWidth, const int imgHeight, 
		const float deltaGrid, const float deltaX, const float deltaY,
		const float xMin, const float yMin, const float zMin,
		const unsigned int voxelGridWidth, const unsigned int voxelGridHeight, const unsigned int voxelGridLength,
		float *offsetX, float *offsetY, float *offsetZ,
		float *r1, float *r2, float *r3, float *r4, float *r5, float *r6, float *r7, float *r8, float *r9,
		unsigned char *char_image_data, unsigned char *voxel_data, unsigned char *hit_counter);

enum NormalType { NORMAL_PER_QUAD, NORMAL_PER_VERTEX};
enum ColorType { COLOR_PER_QUAD, COLOR_PER_VERTEX};

unsigned int computeQuads(float **h_quads, float **h_normals, float **h_colors, 
		unsigned char* d_voxel_grid,
		const unsigned int gridWidth, const unsigned int gridHeight, const unsigned int gridLength,
		const float cube_w, const float cube_h, const float cube_d,
		const unsigned char threshold, 
		const NormalType nt=NORMAL_PER_QUAD, const ColorType ct=COLOR_PER_QUAD);

}

#endif



