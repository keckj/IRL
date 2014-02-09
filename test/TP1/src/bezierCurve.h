
#ifndef _I_BEZIER_CURVE_
#define _I_BEZIER_CURVE_

#include <list> 
#include <GL/glut.h>
#include "parametricCurve.h"

using namespace std;

class BezierCurve : public IParametricCurve
{
	public:
		BezierCurve();
		~BezierCurve();

		void addPoint(GLfloat x, GLfloat y, GLfloat z = 0.0);
		
		GLfloat *getCoords(double t) const;

	private:
		list<GLfloat*> pts;
		static long BinomCoefficient(long n, long k);
	
};

#endif

