#include "voxelViewer.hpp"
#include "voxelViewer.moc"
#include "grid/voxelGrid.hpp"
#include "qtgui/globalGuiVars.hpp"
#include "utils/log.hpp"

#include "memoryManager/sharedResource.hpp"

VoxelViewer::VoxelViewer(float alpha, QWidget *parent) :
	Viewer(parent), alpha(alpha), voxelRenderer(0), boundingBox(0)
{
	qtgui::viewer::viewer = this;

	VoxelGrid<unsigned char,PinnedCPUResource,GPUResource> *grid = qtgui::voxelGrid;

	boundingBox = new BoundingBox(grid->width(), grid->height(), grid->length(),
			alpha*grid->voxelSize());


#ifdef _CUDA_VIEWER 
	voxelRenderer = new VoxelRenderer(grid->width(), grid->height(), grid->length(),
			grid->deviceData(),
			alpha*grid->voxelSize(), alpha*grid->voxelSize(), alpha*grid->voxelSize(),
			&qtgui::viewer::drawViewerBool, &qtgui::viewer::drawOneTime, &qtgui::viewer::viewerThreshold);
#else
	voxelRenderer = new VoxelRenderer(grid->width(), grid->height(), grid->length(),
			grid->dataHost(),
			alpha*grid->voxelSize(), alpha*grid->voxelSize(), alpha*grid->voxelSize(),
			&qtgui::viewer::drawViewerBool, &qtgui::viewer::drawOneTime, &qtgui::viewer::viewerThreshold);
#endif

	this->addRenderable(boundingBox);
	this->addRenderable(voxelRenderer);
}

VoxelViewer::~VoxelViewer() {
	delete voxelRenderer;
	delete boundingBox;
}

void VoxelViewer::keyPressEvent(QKeyEvent* keyEvent) {

	unsigned char &threshold = qtgui::viewer::viewerThreshold;

	switch(keyEvent->key()) {
		case(Qt::Key_Minus): 
			threshold = (threshold > 245 ? 255 : threshold + 10);
			log_console.infoStream() << "Current threshold set to " << (unsigned int) threshold << " !";
			emit updateThreshold();
			drawVoxels();
			break;

		case(Qt::Key_Plus): 
			threshold = (threshold < 10 ? 0 : threshold - 10);
			log_console.infoStream() << "Current threshold set to " << (unsigned int) threshold << " !";
			emit updateThreshold();
			drawVoxels();
			break;
		default:
			emit childKeyEvent(keyEvent);

	}
	
}

void VoxelViewer::drawVoxels() {
	voxelRenderer->computeGeometry();
	this->update();
}

