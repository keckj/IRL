
#ifndef _CYLINDER_
#define _CYLINDER_

#include "renderable.h"
#include <GL/glut.h>

class Cylinder : public Renderable
{
	public:
                Cylinder(int nFaces = 10, double hauteur = 1.0, double rayon = 1.0);
	
				void draw();
                static const float pi = 3.1415;

	private:
                int nFaces;
                double hauteur, rayon;
                GLfloat **base_1;
                GLfloat **base_2;

		void drawImmediate();
};

#endif

