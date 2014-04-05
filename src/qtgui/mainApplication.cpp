
#include "mainApplication.hpp"
#include "qtgui/globalGuiVars.hpp"

MainApplication::MainApplication(VoxelGridTree<unsigned char,PinnedCPUResource,GPUResource> *grid, bool drawVoxels, unsigned char viewerThreshold) 
	: QApplication(0,0), mainWindow(0) {

		qtgui::voxelGrid = grid;
		qtgui::viewer::drawViewerBool = drawVoxels;
		qtgui::viewer::viewerThreshold = viewerThreshold;

		mainWindow = new MainWindow();
}

MainApplication::~MainApplication() {
	delete mainWindow;
}
