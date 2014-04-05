
#ifndef GLOBALGUIVARS_H
#define GLOBALGUIVARS_H

#include "grid/voxelGridTree.hpp"
#include "image/image.hpp"
#include <QObject>
#include <QMainWindow>
#include "memoryManager/PinnedCPUResource.hpp"
#include "memoryManager/GPUResource.hpp"

namespace qtgui {
	
	extern VoxelGridTree<unsigned char,PinnedCPUResource,GPUResource> *voxelGrid;
	extern QMainWindow *mainWindow;

	namespace viewer {
		extern QObject *viewer;
		extern bool drawOneTime;
		extern bool drawViewerBool;
		extern unsigned char viewerThreshold;
	}

	namespace sidepanel {
		extern QObject *sidepanel;
		extern Image::SliceAxe sliceAxe;
		extern unsigned int sliceId;
	}
}

#endif /* end of include guard: GLOBALGUIVARS_H */
