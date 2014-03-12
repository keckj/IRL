
#include "rectangle.hpp"

#include <GL/glut.h>

Rectangle::Rectangle(float width, float height, 
				const float *offset, const float *rotation, const float *color,
				float scale, float postOffsetZ)
: width(width), height(height), offset(offset), rotation(rotation), color(color), scale(scale) {
			
			const float x[] = {0.0f,width,width,0.0f}; 	
			const float y[] = {0.0f,0.0f,height, height}; 	
			
			data = new float[4*3];

			for (int i = 0; i < 4; i++) {
				data[3*i+0] = rotation[0]*x[i] + rotation[1]*y[i] + offset[0];
				data[3*i+1] = rotation[3]*x[i] + rotation[4]*y[i] + offset[1];
				data[3*i+2] = rotation[6]*x[i] + rotation[7]*y[i] + offset[2] + postOffsetZ;
			}

			for (int i = 0; i < 4*3; i++) {
				data[i] *= scale;
			}
}

Rectangle::Rectangle(Rectangle const & other) 
: width(other.width), height(other.height), 
offset(other.offset), rotation(other.rotation), color(other.color), data(0) {
			
			this->data = new float[4*3];
			for (int i = 0; i < 4*3; i++) {
				this->data[i] = other.data[i];
			}
}

Rectangle::~Rectangle() {
	free(data);
}

void Rectangle::draw() {
			glBegin(GL_LINE_LOOP);
			glColor3fv(color);
			for (int i = 0; i < 4; i++) {
				glVertex3fv(data + 3*i);
			}
			glEnd();
}


