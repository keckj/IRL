
#include "grid/voxelGrid.hpp"

namespace qtgui {
	
	VoxelGrid *voxelGrid = 0;

	namespace viewer {
		bool drawViewerBool = true;
		unsigned char viewerThreshold = 127;
	}

	namespace sidepanel {
		unsigned int sliceId = 0;
	}
}
