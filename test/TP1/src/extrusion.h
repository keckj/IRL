

#ifndef _EXTRUSION_
#define _EXTRUSION_

#include "renderable.h"
#include <GL/glut.h>

#include "bezierCurve.h"

class Extrusion : public Renderable
{
	public:
		Extrusion(int nFaces, int nCoupes, BezierCurve const &base, BezierCurve const &generatrice);

		void draw();

	private:
		int nFaces, nCoupes;
		GLfloat ***coupes;

		void drawImmediate();
};

#endif

