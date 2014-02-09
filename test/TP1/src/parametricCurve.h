
#ifndef _I_PARAMETRIC_CURVE_
#define _I_PARAMETRIC_CURVE_

class IParametricCurve
{
	public:
		virtual ~IParametricCurve() {};
		
		//returns coords in [-1,1]^3
		virtual GLfloat *getCoords(double t) const = 0;
	
	private:
		virtual void verify() const {};
};

#endif

