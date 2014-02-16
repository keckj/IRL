
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
	

	unsigned char *data = new unsigned char[256*256*256];

	cout << "INIT" << endl;
	for (int i = 0; i < 256*256*256; i++) {
		data[i] = i%255;
	}

	cout << "DRAW" << endl;
	Viewer viewer;
	viewer.addRenderable(new VoxelRenderer(256, 256, 256, data,
				0.001, 0.001, 0.001, false, 0));
	viewer.setWindowTitle("viewer");
	viewer.show();
	
	// Run main loop.
	application.exec();

	return 0;

	
}
