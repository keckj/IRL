
#ifndef MAINAPPLICATION_H
#define MAINAPPLICATION_H

#include <QApplication>
#include "grid/voxelGridTree.hpp"
#include "qtgui/mainWindow.hpp"
#include "memoryManager/PinnedCPUResource.hpp"
#include "memoryManager/GPUResource.hpp"

class MainApplication : public QApplication {

	public:
		MainApplication(VoxelGridTree<unsigned char,PinnedCPUResource,GPUResource> *grid, bool drawVoxels, unsigned char viewerThreshold);
		~MainApplication();

	private:
		MainWindow *mainWindow;
};

#endif /* end of include guard: MAINAPPLICATION_H */
