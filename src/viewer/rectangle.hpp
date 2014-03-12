
#ifndef RECTANGLE_H

#define RECTANGLE_H

#include "renderable.h"

class Rectangle : public Renderable {
		
	public:
		Rectangle(float width, const float height, 
				const float *offset, const float *rotation, const float *color,
				float scale, float postOffsetZ = 0.0f);
		Rectangle(Rectangle const & other);
		~Rectangle();

		void draw();

	private:
		float width, height;
		const float *offset, *rotation, *color;
		float *data;
		float scale;

};

#endif /* end of include guard: RECTANGLE_H */
