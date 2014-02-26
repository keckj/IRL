
#ifndef _BOUNDING_BOX_
#define _BOUNDING_BOX_

#include <GL/glut.h>
#include "renderable.h"

class BoundingBox : public Renderable
{

public:
	BoundingBox(unsigned int gridWidth, unsigned int gridHeight,
			unsigned int gridLength, float deltaG);

	void draw();

private:
	unsigned int gridWidth, gridHeight, gridLength;
	float deltaG;

};
 #endif
