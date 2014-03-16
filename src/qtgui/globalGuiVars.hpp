
#ifndef GLOBALGUIVARS_H
#define GLOBALGUIVARS_H

#include "grid/voxelGrid.hpp"

namespace qtgui {
	
	extern VoxelGrid *voxelGrid;

	namespace viewer {
		extern bool drawViewerBool;
		extern unsigned char viewerThreshold;
	}

	namespace sidepanel {
		extern unsigned int sliceId;
	}
}

#endif /* end of include guard: GLOBALGUIVARS_H */
