#ifndef _CUBE_
#define _CUBE_

#include "renderable.h"
#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

class Cube : public Renderable
{
	public:
		void draw();

	private:
		void drawImmediate();
		void drawElements();
		void drawArrays();
};

#endif

