#include "voxelRenderer.hpp"
#include <cmath>
#include <GL/glut.h>
#include <iostream>

using namespace std;

VoxelRenderer::VoxelRenderer(int size, float cube_w, float cube_h, float cube_d, bool drawGrid) 
: size(size), cube_w(cube_w), cube_h(cube_h), cube_d(cube_d), drawGrid(drawGrid), tree(size)
{

	for (int x = 0; x < size; x++) {
		//if(x % 10 != 0) 
			//continue;
		for (int y = 0; y < size; y++) {
			for (int z = 0; z < size; z++) {
				tree.set(x,y,z, (x*x*y*z)%255);
			}
		}
	}
}
void VoxelRenderer::draw() {

	//draw grid
	if(drawGrid) {
		drawWireFrame();
	}

	//draw voxels
	Array2D<unsigned char> tmp;
	unsigned char voxel;

	for (int z = 0; z < size; z++ ) {
		tmp = tree.zSlice(z);
		for (int y = 0; y < size; y++ ) {
			for (int x = 0; x < size; x++ ) {
				voxel = tmp(y,x);
				drawVoxel(voxel, x, y, z);
			}
		}
	}
}

void VoxelRenderer::drawVoxel(unsigned char voxel, const int x, const int y, const int z) {

	if(voxel <= 128) {
		return;
	}
	
	glPolygonMode( GL_FRONT, GL_LINE);
	glPolygonMode( GL_BACK, GL_LINE);
	
	float dw, dh, dd;
	dw = 0.5f * cube_w;
	dh = 0.5f * cube_h;
	dd = 0.5f * cube_d;

	glTranslatef(x*cube_w + dw, y*cube_h + dh, z*cube_d + dd);
	glColor3f(voxel/255.0f, voxel/255.0f, voxel/255.0f);

	for (float dx = -dw; dx <= dw; dx+=cube_w) {
		glNormal3f(dx < 0.0f ? -1.0f : 1.0f, 0.0f, 0.0f);
		glBegin(GL_QUADS);
		glVertex3f(dx, dh, dd);
		glVertex3f(dx, -dh, dd);
		glVertex3f(dx, -dh, -dd);
		glVertex3f(dx, dh, -dd);
		glEnd();
	}
	for (float dy = -dh; dy <= dh; dy+=cube_h) {
		glNormal3f(0.0f, dy < 0.0f ? -1.0f : 1.0f, 0.0f);
		glBegin(GL_QUADS);
		glVertex3f(dw, dy, dd);
		glVertex3f(dw, dy, -dd);
		glVertex3f(-dw, dy, -dd);
		glVertex3f(-dw, dy, dd);
		glEnd();
	}
	for (float dz = -dd; dz <= dd; dz+=cube_d) {
		glNormal3f(0.0f, 0.0f, dz < 0.0f ? -1.0f : 1.0f);
		glBegin(GL_QUADS);
		glVertex3f(dw, dh, dz);
		glVertex3f(-dw, dh, dz);
		glVertex3f(-dw, -dh, dz);
		glVertex3f(dw, -dh, dz);
		glEnd();
	}

	glTranslatef(-x*cube_w - dw, -y*cube_h - dh, -z*cube_d - dd);
}

void VoxelRenderer::drawWireFrame() {

	GLfloat *v1,*v2;
	v1 = new GLfloat[3];
	v2 = new GLfloat[3];

	glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	glColor3f(1.0f, 1.0f, 1.0f);

	for (int z = 0; z <= size; z++) {

		for (int y = 0; y <= size; y++) {

			v1[0] = 0; v1[1] = y*cube_h; v1[2] = z*cube_d;
			v2[0] = size*cube_w; v2[1] = y*cube_h; v2[2] = z*cube_d;

			glBegin(GL_LINE_STRIP);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glEnd();
		}

		for (int x = 0; x <= size; x++) {
			v1[0] = x*cube_w; v1[1] = 0; v1[2] = z*cube_d;
			v2[0] = x*cube_w; v2[1] = size*cube_h; v2[2] = z*cube_d;

			glBegin(GL_LINE_STRIP);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glEnd();
		}
	}

	for (int x = 0; x <= size; x++) {
		for (int y = 0; y <= size; y++) {
			v1[0] = x*cube_w; v1[1] = y*cube_h; v1[2] = 0;
			v2[0] = x*cube_w; v2[1] = y*cube_h; v2[2] = size*cube_d;

			glBegin(GL_LINE_STRIP);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glEnd();
		}
	}

}
