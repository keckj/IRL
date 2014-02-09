
#include <iostream>
#include <cmath>
#include "bezierCurve.h"

using namespace std;

BezierCurve::BezierCurve() : pts() {
}

BezierCurve::~BezierCurve() {
}

void BezierCurve::addPoint(GLfloat x, GLfloat y, GLfloat z) {
	GLfloat *pt = new GLfloat[3];
	pt[0] = x;
	pt[1] = y;
	pt[2] = z;

	cout << "Added point " << x << " " << y << " " << z << endl;

	pts.push_back(pt);
}

GLfloat *BezierCurve::getCoords(double t) const {
	unsigned int size = pts.size() - 1;
	
	GLfloat *pt = new GLfloat[3];
	pt[0] = 0.0;
	pt[1] = 0.0;
	pt[2] = 0.0;
	
	unsigned int i = 0;
	for (list<GLfloat*>::const_iterator it = pts.begin(); it != pts.end(); it++) {
		pt[0] += BinomCoefficient(size, i) * (*it)[0] * pow(t, (double) i) * pow(1-t, (double) (size - i));
		pt[1] += BinomCoefficient(size, i) * (*it)[1] * pow(t, (double) i) * pow(1-t, (double) (size - i));
		pt[2] += BinomCoefficient(size, i) * (*it)[2] * pow(t, (double) i) * pow(1-t, (double) (size - i));
		i++;
	}

	return pt;
}

long BezierCurve::BinomCoefficient(long n, long k) {
	if (k > n) { return 0; }
	if (n == k) { return 1; } 
	if (k > n - k) { k = n - k; }
	long c = 1;
	for (long i = 1; i <= k; i++)
	{
		c *= n--;
		c /= i;
	}
	return c;
} 

