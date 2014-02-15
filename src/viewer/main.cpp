
#include <QApplication>
#include <QWidget>
#include "viewer.h"
#include "voxelRenderer.hpp"

//#include "cylinder.h"
using namespace std;

int main(int argc, char** argv)
{
	// Read command lines arguments.
	//
	QApplication application(argc,argv);

	Viewer viewer;
	viewer.addRenderable(new VoxelRenderer(1, 0.01, 0.01, 0.01, false, 127));
	viewer.setWindowTitle("viewer");
	viewer.show();
	
	// Run main loop.
	application.exec();

	cout << "lolo" << endl;
	return 0;

	
}
