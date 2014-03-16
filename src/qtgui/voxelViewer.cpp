#include "voxelViewer.hpp"
#include "grid/voxelGrid.hpp"
#include "qtgui/globalGuiVars.hpp"

VoxelViewer::VoxelViewer(float alpha, QWidget *parent) :
Viewer(parent), alpha(alpha), voxelRenderer(0), boundingBox(0)
{
	
	VoxelGrid *grid = qtgui::voxelGrid;

	boundingBox = new BoundingBox(grid->width(), grid->height(), grid->length(),
			alpha*grid->voxelSize());


#ifdef _CUDA_VIEWER 
	voxelRenderer = new VoxelRenderer(grid->width(), grid->height(), grid->length(),
			grid->dataDevice(),
			alpha*grid->voxelSize(), alpha*grid->voxelSize(), alpha*grid->voxelSize(),
			false, qtgui::viewer::viewerThreshold);
#else
	voxelRenderer = new VoxelRenderer(grid->width(), grid->height(), grid->length(),
			grid->dataHost(),
			alpha*grid->voxelSize(), alpha*grid->voxelSize(), alpha*grid->voxelSize(),
			false, qtgui::viewer::viewerThreshold);
#endif

	this->addRenderable(boundingBox);
	this->addRenderable(voxelRenderer);
}

VoxelViewer::~VoxelViewer() {
	delete voxelRenderer;
	delete boundingBox;
}
