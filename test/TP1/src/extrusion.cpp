#include "extrusion.h"

	Extrusion::Extrusion(int nFaces, int nCoupes, BezierCurve const &base, BezierCurve const &generatrice) 
: nFaces(nFaces), nCoupes(nCoupes)
{

	double dz = 1.0/nCoupes;
	double dt = 1.0/nFaces;
	double t, z;

	GLfloat *buffer;

	coupes = new GLfloat**[nCoupes];

	z = 0.0;
	for (int nC = 0; nC < nCoupes; nC++) {
		coupes[nC] = new GLfloat*[nFaces];
		t = 0.0;
		for (int nF = 0; nF < nFaces; nF++) {
			coupes[nC][nF] = generatrice.getCoords(z);
			buffer = base.getCoords(t);
			coupes[nC][nF][0] += buffer[0]; 
			coupes[nC][nF][1] += buffer[1];
			coupes[nC][nF][2] += buffer[2];
			t += dt;
		}
		z += dz;
	}
}

void Extrusion::draw() {
	glTranslatef(1.0,0,0);
	drawImmediate();
}

void Extrusion::drawImmediate() {

	GLfloat *normal = new GLfloat[3];

	//Face du bas  
	normal[0] = 0.0; normal[1] = 0.0; normal[2] = -1.0;

	glBegin(GL_POLYGON); 
	glColor3f(0.6,0.0,0.6);
	glNormal3fv(normal);
	for (int i = 0; i < nFaces; i++) {
		glVertex3fv(coupes[0][i]);
		printf("Point : %f, %f, %f\n",  coupes[0][i][0], coupes[0][i][1], coupes[0][i][2]);
	}
	glEnd();

	//Face du haut
	normal[0] = 0.0; normal[1] = 0.0; normal[2] = +1.0;

	glBegin(GL_POLYGON); 
	glColor3f(0.1,0.0,0.6);
	glNormal3fv(normal);
	for (int i = 0; i < nFaces; i++) {
		glVertex3fv(coupes[nCoupes-1][i]);
	}
	glEnd();

	//Bords
	GLfloat **coupe_1, **coupe_2;
	glNormal3fv(normal);
	
	for (int nC = 1; nC < nCoupes; nC++) {
		coupe_1 = coupes[nC-1];
		coupe_2 = coupes[nC]; 
		
		glColor3f(0.1 + 0.5*(1.0 - (double)nC/nCoupes), 0.0 ,0.6);
		glBegin(GL_QUAD_STRIP); 
		
		for (int nF = 0; nF <= nFaces; nF++) {
			glVertex3fv(coupe_1[nF%nFaces]);
			glVertex3fv(coupe_2[nF%nFaces]);
		}
		glEnd();
	}
}

