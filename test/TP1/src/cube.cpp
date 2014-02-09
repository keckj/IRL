#include <iostream>
using namespace std;
#include "cube.h"

/*
          Z
          |
    7 +-------+ 6
     /|      /|
    / |     / |   ---> Y
 4 +-------+5 |
   | 3+----|--+ 2
   | /     | /
   |/      |/
 0 +-------+ 1
      /
     X         
*/


void Cube::draw()
{
	glPushMatrix();

	// draw immediate (center cube)
	drawImmediate();

	// draw arrays (left cube)
	glTranslatef(-2, 0, 0);
	drawArrays();

	// draw elements (right cube)
	glTranslatef(+4, 0, 0);
	drawElements();

	glPopMatrix();
}


//==================== 1. Immediate method ===================================
// immediate definition of individual vertex properties

void Cube::drawImmediate()
{
	// vertices coordinates can be stored
//	GLfloat s0[] = {+0.5, -0.5, -0.5};
	GLfloat s1[] = {+0.5, +0.5, -0.5};
	GLfloat s2[] = {-0.5, +0.5, -0.5};
	GLfloat s3[] = {-0.5, -0.5, -0.5};
	GLfloat s4[] = {+0.5, -0.5, +0.5};
	GLfloat s5[] = {+0.5, +0.5, +0.5};
	GLfloat s6[] = {-0.5, +0.5, +0.5};
	GLfloat s7[] = {-0.5, -0.5, +0.5};
	
	
	// For each vertex of each face, define its normal then its coordinates
	// - vertices shared by faces must be reset (glVertex3f) for each face
	// - vertices of the same polygon might share the same normal (flat
	//   surface) or have different normals (curved surface)
	
	glBegin(GL_QUADS);

		// for each face, define a normal (first) the a 
		// 0 3 2 1
		glNormal3f(0.0, 0.0, -1.0);	// same normal shared by 4 vertices
		glVertex3f(+0.5, -0.5, -0.5);   // direct coordinates
		glVertex3fv(s1);                // stored coordinates
		glVertex3fv(s2);
		glVertex3fv(s3);

		// 0 4 7 3
		glNormal3f(0.0, -1.0, 0.0);
		glVertex3f(+0.5, -0.5, -0.5);
		glVertex3fv(s4);
		glVertex3fv(s7);
		glVertex3fv(s3);

		// 1 5 4 0
		glNormal3f(1.0, 0.0, 0.0);
		glVertex3fv(s1);
		glVertex3fv(s5);
		glVertex3fv(s4);
		glVertex3f(+0.5, -0.5, -0.5);

		// 2 6 5 1
		glNormal3f(0.0, 1.0, 0.0);
		glVertex3fv(s2);
		glVertex3fv(s6);
		glVertex3fv(s5);
		glVertex3fv(s1);
		
		// 3 7 6 2
		glNormal3f(-1.0, 0.0, 0.0);
		glVertex3fv(s3);
		glVertex3fv(s7);
		glVertex3fv(s6);
		glVertex3fv(s2);
		
		// 4 5 6 7
		glNormal3f(0.0, 0.0, 1.0);
		glVertex3fv(s4);
		glVertex3fv(s5);
		glVertex3fv(s6);
		glVertex3fv(s7);

	glEnd();
}




//==================== 2. Arrays - drawArrays ================================
// - store vertex-related data (coordinates, normals, colors, etc) in arrays
//   => reduces the number of function calls

// Define the coordinates of all vertices, ordered face by face.
// Since a vertex is shared by 3 faces, it is repeated 3 times!
static GLfloat allVertices[] = { 
	+0.5, -0.5, -0.5,   -0.5, -0.5, -0.5,   -0.5, +0.5, -0.5,   +0.5, +0.5, -0.5,  // 0 3 2 1
	+0.5, -0.5, -0.5,   +0.5, -0.5, +0.5,   -0.5, -0.5, +0.5,   -0.5, -0.5, -0.5,  // 0 4 7 3
	+0.5, +0.5, -0.5,   +0.5, +0.5, +0.5,   +0.5, -0.5, +0.5,   +0.5, -0.5, -0.5,  // 1 5 4 0
	-0.5, +0.5, -0.5,   -0.5, +0.5, +0.5,   +0.5, +0.5, +0.5,   +0.5, +0.5, -0.5,  // 2 6 5 1
	-0.5, -0.5, -0.5,   -0.5, -0.5, +0.5,   -0.5, +0.5, +0.5,   -0.5, +0.5, -0.5,  // 3 7 6 2
	+0.5, -0.5, +0.5,   +0.5, +0.5, +0.5,   -0.5, +0.5, +0.5,   -0.5, -0.5, +0.5,  // 4 5 6 7 
};

// Define the normals of vertices, using the same vertex order.
static GLfloat allNormals[] = { 
	 0.0,  0.0, -1.0,    0.0,  0.0, -1.0,    0.0,  0.0, -1.0,    0.0,  0.0, -1.0,  // 0 3 2 1
	 0.0, -1.0,  0.0,    0.0, -1.0,  0.0,    0.0, -1.0,  0.0,    0.0, -1.0,  0.0,  // 0 4 7 3
	 1.0,  0.0,  0.0,    1.0,  0.0,  0.0,    1.0,  0.0,  0.0,    1.0,  0.0,  0.0,  // 1 5 4 0
	 0.0,  1.0,  0.0,    0.0,  1.0,  0.0,    0.0,  1.0,  0.0,    0.0,  1.0,  0.0,  // 2 6 5 1
	-1.0,  0.0,  0.0,   -1.0,  0.0,  0.0,   -1.0,  0.0,  0.0,   -1.0,  0.0,  0.0,  // 3 7 6 2
	 0.0,  0.0,  1.0,    0.0,  0.0,  1.0,    0.0,  0.0,  1.0,    0.0,  0.0,  1.0,  // 4 5 6 7
};


void Cube::drawArrays()
{	
	// activate the use of GL_VERTEX_ARRAY and GL_NORMAL_ARRAY
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);
	
	// specify the arrays to use
	glNormalPointer(GL_FLOAT, 0, allNormals);
	glVertexPointer(3, GL_FLOAT, 0 , allVertices);
	
	// draw quads using 24 data stored in arrays, starting at index 0
	glDrawArrays(GL_QUADS, 0, 24);
	
	// disable the use of arrays (do not forget!)
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	
	// Others arrays could also be used: 
	// GL_COLOR_ARRAY, GL_INDEX_ARRAY, GL_TEXTURE_COORD_ARRAY, ...
}



//==================== 3. Arrays - drawElements ==============================
// - single definition of shared data
// - draw face by face, using face indices

// vertex coordinates (defined once only)
static GLfloat vertices[] = { 
	+0.5, -0.5, -0.5,  // 0
	+0.5, +0.5, -0.5,  // 1
	-0.5, +0.5, -0.5,  // 2
	-0.5, -0.5, -0.5,  // 3
	+0.5, -0.5, +0.5,  // 4
	+0.5, +0.5, +0.5,  // 5
	-0.5, +0.5, +0.5,  // 6
	-0.5, -0.5, +0.5,  // 7
};

// indexes of the 6 faces (in a 2D array)
static GLubyte indices[6][4] = {
	{0, 3, 2, 1},
	{0, 4, 7, 3},
	{1, 5, 4, 0},
	{2, 6, 5, 1},
	{3, 7, 6, 2},
	{4, 5, 6, 7}
};

// normals (defined once only)
static GLfloat normals[6][3] = {
	{ 0.0,  0.0, -1.0},
	{ 0.0, -1.0,  0.0},
	{ 1.0,  0.0,  0.0},
	{ 0.0,  1.0,  0.0},
	{-1.0,  0.0,  0.0},
	{ 0.0,  0.0,  1.0}
};


void Cube::drawElements()
{
	// activate the use of GL_VERTEX_ARRAY (not GL_NORMALS_ARRAY)
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0 , vertices);
	
	// for each face: 
	//   - set the current normal (state machine)
	//   - draw a face using glDrawElements with the indices of the face
	
	glNormal3fv(normals[0]);
	glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices[0]);
	glNormal3fv(normals[1]);
	glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices[1]);
	glNormal3fv(normals[2]);
	glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices[2]);
	glNormal3fv(normals[3]);
	glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices[3]);
	glNormal3fv(normals[4]);
	glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices[4]);
	glNormal3fv(normals[5]);
	glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices[5]);

	glDisableClientState(GL_VERTEX_ARRAY);
}



