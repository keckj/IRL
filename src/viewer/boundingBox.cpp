#include "boundingBox.hpp"
#include <iostream>

BoundingBox::BoundingBox(unsigned int gridWidth, unsigned int gridHeight, unsigned int gridLength, 
		float deltaG,
		float offsetX, float offsetY, float offsetZ, 
		float r, float g, float b) :
	gridWidth(gridWidth), gridHeight(gridHeight), gridLength(gridLength), 
	deltaG(deltaG),
	offsetX(offsetX), offsetY(offsetY), offsetZ(offsetZ),
	_r(r), _g(g), _b(b)
{
}

void BoundingBox::draw() {
			
	GLfloat *v1,*v2;
	v1 = new GLfloat[3];
	v2 = new GLfloat[3];

	glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	glColor3f(_r, _g, _b);

	for (unsigned int z = 0; z <= 1; z++) {
		for (unsigned int y = 0; y <= 1; y++) {

			v1[0] = offsetX + 0.0f; v1[1] = offsetY + y*gridHeight*deltaG; v1[2] = offsetZ + z*gridLength*deltaG;
			v2[0] = offsetX + deltaG*gridWidth; v2[1] = offsetY + y*gridHeight*deltaG; v2[2] = offsetZ + z*gridLength*deltaG;

			glBegin(GL_LINE_STRIP);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glEnd();
		}

		for (unsigned int x = 0; x <= 1; x++) {
			v1[0] = offsetX + x*gridWidth*deltaG; v1[1] = offsetY + 0.0f; v1[2] = offsetZ + z*gridLength*deltaG;
			v2[0] = offsetX + x*gridWidth*deltaG; v2[1] = offsetY + deltaG*gridHeight; v2[2] = offsetZ + z*gridLength*deltaG;

			glBegin(GL_LINE_STRIP);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glEnd();
		}
	}

	for (unsigned int x = 0; x <= 1; x++) {
		for (unsigned int y = 0; y <= 1; y++) {
			v1[0] = offsetX + x*gridWidth*deltaG; v1[1] = offsetY + y*gridHeight*deltaG; v1[2] = offsetZ + 0.0f;
			v2[0] = offsetX + x*gridWidth*deltaG; v2[1] = offsetY + y*gridHeight*deltaG; v2[2] = offsetZ + gridLength*deltaG;

			glBegin(GL_LINE_STRIP);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glEnd();
		}
	}
}
