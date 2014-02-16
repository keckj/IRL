
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
	

	unsigned char *data = new unsigned char[10*10*10];

	Viewer viewer;
	viewer.addRenderable(new VoxelRenderer(10, 10, 10, data,
				0.01, 0.01, 0.01, false, 127));
	viewer.setWindowTitle("viewer");
	viewer.show();
	
	// Run main loop.
	application.exec();

	cout << "lolo" << endl;
	return 0;

	
}
