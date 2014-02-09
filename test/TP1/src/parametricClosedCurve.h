
#ifndef _I_PARAMETRIC_CLOSED_CURVE_
#define _I_PARAMETRIC_CLOSED_CURVE_

#include "parametricCurve.h"

class IParametricClosedCurve : public IParametricCurve {
	
	private:
		virtual void verify() const {
			assert(getCoords(0) = getCoords(1));
		}

};

#endif
