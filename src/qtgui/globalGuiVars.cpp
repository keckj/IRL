
#include "grid/voxelGridTree.hpp"
#include "image/image.hpp"
#include <QObject>
#include <QMainWindow>
#include "memoryManager/PinnedCPUResource.hpp"
#include "memoryManager/GPUResource.hpp"

namespace qtgui {
	
	VoxelGridTree<unsigned char,PinnedCPUResource,GPUResource> *voxelGrid = 0;
	QMainWindow *mainWindow = 0;

	namespace viewer {
		QObject *viewer = 0;
		bool drawOneTime = false;
		bool drawViewerBool = true;
		unsigned char viewerThreshold = 127;
	}


	namespace sidepanel {
		QObject *sidepanel = 0;
		unsigned int sliceId = 0;
		Image::SliceAxe sliceAxe = Image::SliceAxe::AXE_X;
	}
}
