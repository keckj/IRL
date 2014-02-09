
#include <cmath>
#include <cylinder.h>


Cylinder::Cylinder(int nFaces, double hauteur, double rayon) {
        this->nFaces = nFaces;
        this->hauteur = hauteur;
        this->rayon = rayon;

        this->base_1 = new GLfloat*[nFaces];
        this->base_2 = new GLfloat*[nFaces];

        double theta = 0.0;
        for (int i = 0; i < nFaces; i++) {
                base_1[i] = new GLfloat[3];
                base_1[i][0] = rayon*cos(theta);
                base_1[i][1] = rayon*sin(theta);
                base_1[i][2] = 0;

                base_2[i] = new GLfloat[3];
                base_2[i][0] = rayon*cos(theta);
                base_2[i][1] = rayon*sin(theta);
                base_2[i][2] = hauteur;
                
                theta += 2.0*pi/nFaces;
        }
}

void Cylinder::draw() {

	glPushMatrix();

	// draw immediate (center cube)
	drawImmediate();

	glPopMatrix();
}

void Cylinder::drawImmediate() {

        GLfloat *normal = new GLfloat[3];

        //Face du bas  
        normal[0] = 0.0; normal[1] = 0.0; normal[2] = -1.0;

        glBegin(GL_POLYGON); 
                glColor3f(1.0,0.0,0.0);
                glNormal3fv(normal);
                for (int i = 0; i < nFaces; i++) {
                        glVertex3fv(base_1[i]);
                }
        glEnd();
        
        //Face du haut
        normal[0] = 0.0; normal[1] = 0.0; normal[2] = +1.0;

        glBegin(GL_POLYGON); 
                glColor3f(0.0,1.0,0.0);
                glNormal3fv(normal);
                for (int i = 0; i < nFaces; i++) {
                        glVertex3fv(base_2[i]);
                }
        glEnd();

        //Bord avec vraies normales
        glBegin(GL_QUAD_STRIP); 
                glColor3f(0.0,0.0,1.0);
                glNormal3fv(normal);
                for (int i = 0; i <= nFaces; i++) {
                        glVertex3fv(base_1[i%nFaces]);
                        glVertex3fv(base_2[i%nFaces]);
                }
        glEnd();

}

