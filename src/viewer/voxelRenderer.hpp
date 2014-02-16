
#ifndef _VOXEL_RENDERER_
#define _VOXEL_RENDERER_

#include "voxel.hpp"

class VoxelRenderer : public Renderable
{
	public:

		VoxelRenderer( 
				unsigned int width, unsigned int height, unsigned int length, 
				unsigned char *data,
				float cube_w, float cube_h, float cube_d, 
				bool drawGrid, unsigned char threshold);

		void draw();

	private:
		unsigned int width, height, length;
		unsigned char *data;
		float cube_w, cube_h, cube_d;
		bool drawGrid;
		unsigned char threshold;

		enum Side { UP, DOWN, LEFT, RIGHT, FRONT, BACK };		

		void drawSurface();
		void drawNaive();
		void drawWireFrame();


		unsigned char getData(unsigned int x, unsigned int y, unsigned int z);

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

