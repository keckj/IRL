
#ifndef _BOUNDING_BOX_
#define _BOUNDING_BOX_

#include <GL/glut.h>
#include "renderable.h"

class BoundingBox : public Renderable
{

public:
	BoundingBox(
			unsigned int gridWidth, unsigned int gridHeight, unsigned int gridLength, 
			float deltaG,
			float offsetX = 0.0f, float offsetY = 0.0f, float offsetZ = 0.0f, 
			float _r = 0.0f, float _g = 0.0f, float _b = 1.0f);


	void draw();

private:
	unsigned int gridWidth, gridHeight, gridLength;
	float deltaG;
	float offsetX, offsetY, offsetZ;
	float _r, _g, _b;

};
 #endif
