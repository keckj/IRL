
#ifndef _VOXEL_RENDERER_
#define _VOXEL_RENDERER_

#include "octree.h"
#include "voxel.hpp"

class VoxelRenderer : public Renderable
{
	public:
		VoxelRenderer(int size, float cube_w = 1.0f, float cube_h = 1.0f, float cube_d = 1.0f, bool drawGrid = true, unsigned char threshold = 128);	
		void draw();

	private:
		int size;
		float cube_w, cube_h, cube_d;
		bool drawGrid;
		Octree<unsigned char> tree;
		unsigned char threshold;
	
		enum Side { UP, DOWN, LEFT, RIGHT, FRONT, BACK };		

		void drawSurface();
		void drawNaive();
		void drawWireFrame();
		
		bool inline isVisible(unsigned char voxel);

		void inline drawVoxel(
				unsigned char voxel,
				int x, int y, int z);

		void inline drawVoxel(
				unsigned char current, 
				unsigned char right, unsigned char left, 
				unsigned char up, unsigned char down, 
				unsigned char front, unsigned char back,
				int x, int y, int z);
		
		void inline drawQuad(Side side);

};

#endif

