
#ifndef _VOXEL_RENDERER_
#define _VOXEL_RENDERER_

#include "octree.h"
#include "voxel.hpp"

class VoxelRenderer : public Renderable
{
	public:
		VoxelRenderer(int size, float cube_w = 1.0f, float cube_h = 1.0f, float cube_d = 1.0f, bool drawGrid = true);	
		void draw();

	private:
		int size;
		float cube_w, cube_h, cube_d;
		bool drawGrid;
		Octree<unsigned char> tree;

		void drawWireFrame();
		void drawVoxel(unsigned char voxel, int x, int y, int z);
};

#endif

