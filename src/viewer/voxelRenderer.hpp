
#ifndef _VOXEL_RENDERER_
#define _VOXEL_RENDERER_

#define _CUDA_VIEWER

#include "voxel.hpp"
#include <GL/glut.h>

class VoxelRenderer : public Renderable
{
	public:

		VoxelRenderer( 
				unsigned int width, unsigned int height, unsigned int length, 
				unsigned char *data,
				float cube_w, float cube_h, float cube_d, 
				bool drawGrid, unsigned char threshold);

		void computeGeometry();
		void draw();
	
	
		void keyPressEvent(QKeyEvent*, Viewer&);

	private:
		unsigned int width, height, length;
		unsigned char *data;
		float cube_w, cube_h, cube_d;
		bool drawGrid;
		unsigned char threshold;

		unsigned int nQuads;
		GLfloat *quads, *normals, *colors;

		enum Side { UP, DOWN, LEFT, RIGHT, FRONT, BACK };		

		void drawWireFrame();

		unsigned char getData(unsigned int x, unsigned int y, unsigned int z);

		bool inline isVisible(unsigned char voxel);

		void writeVectAndIncr(GLfloat *array, unsigned int &offset, GLfloat x, GLfloat y, GLfloat z);
		void writeVect(GLfloat *array, unsigned int offset, GLfloat x, GLfloat y, GLfloat z);


		unsigned int inline countQuads();

		unsigned char inline countFaces(
				unsigned char current, 
				unsigned char right, unsigned char left, 
				unsigned char up, unsigned char down, 
				unsigned char front, unsigned char back);

		void inline computeQuadsAndNormals();

		void inline computeVoxel(
				unsigned char current, 
				unsigned char right, unsigned char left, 
				unsigned char up, unsigned char down, 
				unsigned char front, unsigned char back,
				int x, int y, int z, 
				unsigned int& offset);

		void inline computeQuads(
				Side side, unsigned char current, 
				unsigned int &offset,
				int x, int y, int z);

};
#endif

