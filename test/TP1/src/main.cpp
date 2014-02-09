
#include <QApplication>
#include <QWidget>
#include "viewer.h"
#include "cylinder.h"
#include "bezierCurve.h"
#include "extrusion.h"
#include "voxelRenderer.hpp"

//#include "cylinder.h"
using namespace std;

int main(int argc, char** argv)
{
	// Read command lines arguments.
	//
	QApplication application(argc,argv);
	
	
	//QWidget window;

	//window.resize(250, 150);
	//window.setWindowTitle("Simple example");
	//window.show();

	//BezierCurve base, generatrice;

	//generatrice.addPoint(0.0, 0.0, 0.0);
	//generatrice.addPoint(1.0, 0.0, 1.0);
	//generatrice.addPoint(1.0, 1.0, 2.0);
	//generatrice.addPoint(-1.0, 1.0, 3.0);
	//generatrice.addPoint(-1.0, 0.0, 4.0);
	//generatrice.addPoint(-1.0, -1.0, 5.0);

	//base.addPoint(0.5, 0.5, 0.0);
	//base.addPoint(-0.5, 0.5, 0.0);
	//base.addPoint(-0.5, -0.5, 0.0);
	//base.addPoint(0.5, -0.5, 0.0);
	//base.addPoint(0.5, 0.5, 0.0);

	Viewer viewer;
	viewer.addRenderable(new VoxelRenderer(32, 4.0, 2.0, 1.0, false));
	viewer.setWindowTitle("viewer");
	viewer.show();
	
	// Run main loop.
	return application.exec();
}
