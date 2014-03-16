#include "voxelRenderer.hpp"
#include <cmath>
#include <GL/glut.h>
#include <iostream>
#include "utils/log.hpp"
#include "kernels/kernels.hpp"
#include "utils/cudaUtils.hpp"

using namespace std;

VoxelRenderer::VoxelRenderer( 
		unsigned int width, unsigned int height, unsigned int length, 
		unsigned char *data,
		float cube_w, float cube_h, float cube_d, 
		bool drawGrid, unsigned char threshold) 
:	width(width), height(height), length(length), data(data),
	cube_w(cube_w), cube_h(cube_h), cube_d(cube_d), 
	drawGrid(drawGrid), threshold(threshold),
	nQuads(0), quads(0), normals(0), colors(0)
{
#ifdef _CUDA_VIEWER
	log_console.info("Created a CUDA Voxel Renderer !");
#else
	log_console.info("Created a CPU Voxel Renderer !");
#endif
	
	log_console.infoStream() << "Current threshold set to " << (unsigned int) threshold << " !";

	computeGeometry();
}

void VoxelRenderer::draw() {

	//draw grid
	if(drawGrid) {
		drawWireFrame();
	}

	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL);
	
	// activate the use of GL_VERTEX_ARRAY, GL_NORMAL_ARRAY and GL_COLOR_ARRAY
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	// specify the arrays to use
	glNormalPointer(GL_FLOAT, 0, normals);
	glVertexPointer(3, GL_FLOAT, 0 , quads);
	glColorPointer(3, GL_FLOAT, 0 , colors);
	
	//draw quads
	glDrawArrays(GL_QUADS, 0, 4*nQuads);

	// disable the use of arrays (do not forget!)
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
		
}

bool inline VoxelRenderer::isVisible(unsigned char voxel) {
	return (voxel > threshold);
}

unsigned char VoxelRenderer::getData(unsigned int x, unsigned int y, unsigned int z) {
	return data[z*width*height + y*width + x];
}

void VoxelRenderer::drawWireFrame() {

	GLfloat *v1,*v2;
	v1 = new GLfloat[3];
	v2 = new GLfloat[3];

	glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	glColor3f(1.0f, 1.0f, 1.0f);

	for (unsigned int z = 0; z <= length; z++) {

		for (unsigned int y = 0; y <= height; y++) {

			v1[0] = 0; v1[1] = y*cube_h; v1[2] = z*cube_d;
			v2[0] = width*cube_w; v2[1] = y*cube_h; v2[2] = z*cube_d;

			glBegin(GL_LINE_STRIP);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glEnd();
		}

		for (unsigned int x = 0; x <= width; x++) {
			v1[0] = x*cube_w; v1[1] = 0; v1[2] = z*cube_d;
			v2[0] = x*cube_w; v2[1] = height*cube_h; v2[2] = z*cube_d;

			glBegin(GL_LINE_STRIP);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glEnd();
		}
	}

	for (unsigned int x = 0; x <= width; x++) {
		for (unsigned int y = 0; y <= height; y++) {
			v1[0] = x*cube_w; v1[1] = y*cube_h; v1[2] = 0;
			v2[0] = x*cube_w; v2[1] = y*cube_h; v2[2] = length*cube_d;

			glBegin(GL_LINE_STRIP);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glEnd();
		}
	}



}


void VoxelRenderer::computeGeometry() {

//CUDA based quad generation
#ifdef _CUDA_VIEWER
	cudaFreeHost(quads);
	cudaFreeHost(colors);
	cudaFreeHost(normals);
	
	
	nQuads = kernel::computeQuads(
			&quads, &normals, &colors, 
			data,
			width, height, length,
			cube_w, cube_h, cube_d,
			threshold,
			kernel::NORMAL_PER_VERTEX, kernel::COLOR_PER_VERTEX);

//CPU based quad generation
#else
	delete [] quads;
	delete [] normals;
	delete [] colors;
	nQuads = countQuads();
	quads = new GLfloat[3*4*nQuads];
	normals = new GLfloat[3*nQuads];
	colors = new GLfloat[3*nQuads];
	computeQuadsAndNormals();
#endif
	
	log_console.infoStream() << "Computed " << nQuads << " quads !";
}


unsigned int inline VoxelRenderer::countQuads() {
	unsigned int counter = 0;
	unsigned char left, right, up, down, front, back, current;

	for (unsigned int z = 0; z < length; z++ ) {
		for (unsigned int y = 0; y < height; y++ ) {
			for (unsigned int x = 0; x < width; x++ ) {
				right = (x == width - 1 ? 0 : getData(x+1,y,z));
				left = (x == 0 ? 0 : getData(x-1,y,z));

				up = (y == height - 1 ? 0 : getData(x,y+1,z));
				down = (y == 0 ? 0 : getData(x,y-1,z));

				front = (z == length - 1 ? 0 : getData(x,y,z+1));
				back = (z == 0 ? 0 : getData(x,y,z-1));

				current = getData(x,y,z);	
				counter += countFaces(current, right, left, up, down, front, back);
			}
		}
	}

	return counter;
}

unsigned char inline VoxelRenderer::countFaces (
		unsigned char current, 
		unsigned char right, unsigned char left, 
		unsigned char up, unsigned char down, 
		unsigned char front, unsigned char back
		) {

	if(!isVisible(current))	
		return 0;

	return (!isVisible(up)) + (!isVisible(down)) + (!isVisible(right)) +
		(!isVisible(left)) + (!isVisible(front)) + (!isVisible(back));
}

void inline VoxelRenderer::computeQuadsAndNormals() {

	unsigned char left, right, up, down, front, back, current;

	unsigned int offset = 0;

	for (unsigned int z = 0; z < length; z++ ) {
		for (unsigned int y = 0; y < height; y++ ) {
			for (unsigned int x = 0; x < width; x++ ) {
				right = (x == width - 1 ? 0 : getData(x+1,y,z));
				left = (x == 0 ? 0 : getData(x-1,y,z));

				up = (y == height - 1 ? 0 : getData(x,y+1,z));
				down = (y == 0 ? 0 : getData(x,y-1,z));

				front = (z == length - 1 ? 0 : getData(x,y,z+1));
				back = (z == 0 ? 0 : getData(x,y,z-1));

				current = getData(x,y,z);	

				computeVoxel(current, right, left, up, down, front, back, x,y,z, offset);
			}
		}
	}

}


void inline VoxelRenderer::computeVoxel(
		unsigned char current, 
		unsigned char right, unsigned char left, 
		unsigned char up, unsigned char down, 
		unsigned char front, unsigned char back,
		int x, int y, int z, 
		unsigned int &offset) {

	if(!isVisible(current))	
		return;


	if(!isVisible(up))
		computeQuads(UP, current, offset, x, y, z);

	if(!isVisible(down))
		computeQuads(DOWN, current, offset, x, y, z);

	if(!isVisible(right))
		computeQuads(RIGHT, current, offset, x, y, z);

	if(!isVisible(left))
		computeQuads(LEFT, current, offset, x, y, z);

	if(!isVisible(front))
		computeQuads(FRONT, current, offset, x, y, z);

	if(!isVisible(back))
		computeQuads(BACK, current, offset, x, y, z);

}

void inline VoxelRenderer::computeQuads(
		Side side, unsigned char current, 
		unsigned int &offset,
		int x, int y, int z) {

	const float tx = x*cube_w;
	const float ty = y*cube_h;
	const float tz = z*cube_d;

	writeVect(colors, offset/4, current/255.0f, current/255.0f, current/255.0f);

	switch(side) 
	{
		case UP: 
			{
				writeVect(normals, offset/4, 0.0f, 1.0f, 0.0f);
				writeVectAndIncr(quads, offset, 0.0f + tx, cube_h + ty, 0.0f + tz);
				writeVectAndIncr(quads, offset, 0.0f + tx, cube_h + ty, cube_d + tz);
				writeVectAndIncr(quads, offset, cube_w + tx, cube_h + ty, cube_d + tz);
				writeVectAndIncr(quads, offset, cube_w + tx, cube_h + ty, 0.0f + tz);
				break;
			}

		case DOWN: 
			{
				writeVect(normals, offset/4, 0.0f, -1.0f, 0.0f);
				writeVectAndIncr(quads, offset, 0.0f + tx, 0.0f + ty, 0.0f + tz);
				writeVectAndIncr(quads, offset, 0.0f + tx, 0.0f + ty, cube_d + tz);
				writeVectAndIncr(quads, offset, cube_w + tx, 0.0f + ty, cube_d + tz);
				writeVectAndIncr(quads, offset, cube_w + tx, 0.0f + ty, 0.0f + tz);
				break;
			}

		case RIGHT: 
			{
				writeVect(normals, offset/4, 1.0f, 0.0f, 0.0f);
				writeVectAndIncr(quads, offset, cube_w + tx, 0.0f + ty, 0.0f + tz);
				writeVectAndIncr(quads, offset, cube_w + tx, cube_h + ty, 0.0f + tz);
				writeVectAndIncr(quads, offset, cube_w + tx, cube_h + ty, cube_d + tz);
				writeVectAndIncr(quads, offset, cube_w + tx, 0.0f + ty, cube_d + tz);
				break;
			}
		case LEFT: 
			{
				writeVect(normals, offset/4, -1.0f, 0.0f, 0.0f);
				writeVectAndIncr(quads, offset, 0.0f + tx, 0.0f + ty, 0.0f + tz);
				writeVectAndIncr(quads, offset, 0.0f + tx, cube_h + ty, 0.0f + tz);
				writeVectAndIncr(quads, offset, 0.0f + tx, cube_h + ty, cube_d + tz);
				writeVectAndIncr(quads, offset, 0.0f + tx, 0.0f + ty, cube_d + tz);
				break;
			}
		case FRONT: 
			{
				writeVect(normals, offset/4, 0.0f, 0.0f, 1.0f);
				writeVectAndIncr(quads, offset, 0.0f + tx, 0.0f + ty, cube_d + tz);
				writeVectAndIncr(quads, offset, cube_w + tx, 0.0f + ty, cube_d + tz);
				writeVectAndIncr(quads, offset, cube_w + tx, cube_h + ty, cube_d + tz);
				writeVectAndIncr(quads, offset, 0.0f + tx, cube_h + ty, cube_d + tz);
				break;
			}
		case BACK: 
			{
				writeVect(normals, offset/4, 0.0f, 0.0f, -1.0f);
				writeVectAndIncr(quads, offset, 0.0f + tx, 0.0f + ty, 0.0f + tz);
				writeVectAndIncr(quads, offset, cube_w + tx, 0.0f + ty, 0.0f + tz);
				writeVectAndIncr(quads, offset, cube_w + tx, cube_h + ty, 0.0f + tz);
				writeVectAndIncr(quads, offset, 0.0f + tx, cube_h + ty, 0.0f + tz);
				break;
			}
	}

}

void VoxelRenderer::writeVectAndIncr(GLfloat *array, unsigned int &offset, GLfloat x, GLfloat y, GLfloat z) {
	array[offset++] = x;
	array[offset++] = y;
	array[offset++] = z;
}

void VoxelRenderer::writeVect(GLfloat *array, unsigned int offset, GLfloat x, GLfloat y, GLfloat z) {
	unsigned int tmp = offset;
	array[tmp++] = x;
	array[tmp++] = y;
	array[tmp++] = z;
}
		
void VoxelRenderer::keyPressEvent(QKeyEvent* keyEvent, Viewer& viewer) {
	switch(keyEvent->key()) {
		case(Qt::Key_Minus): {
			threshold = (threshold > 245 ? 255 : threshold + 10);
			log_console.infoStream() << "Current threshold set to " << (unsigned int) threshold << " !";
			computeGeometry();
			break;
		}

		case(Qt::Key_Plus): {
			threshold = (threshold < 10 ? 0 : threshold - 10);
			log_console.infoStream() << "Current threshold set to " << (unsigned int) threshold << " !";
			computeGeometry();
			break;
		}
	}
}
