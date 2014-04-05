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

	VoxelGridTree<unsigned char,PinnedCPUResource,GPUResource> *grid = qtgui::voxelGrid;

	voxelRenderer = new VoxelRenderer(grid, alpha, 
			&qtgui::viewer::drawViewerBool, &qtgui::viewer::drawOneTime, &qtgui::viewer::viewerThreshold);
	
	boundingBox = new BoundingBox(grid->width(), grid->height(), grid->length(),
			alpha*grid->voxelSize());

	unsigned int gridIdx = 0, gridIdy = 0,  gridIdz = 0;
	float dl = alpha*grid->voxelSize();
	for (unsigned int i = 0; i < grid->nChilds(); i++) {

			this->addRenderable(
					new BoundingBox(
						grid->subwidth(), grid->subheight(), grid->sublength(), 
						dl,
						gridIdx*grid->subwidth()*dl, gridIdy*grid->subheight()*dl, gridIdz*grid->sublength()*dl,
						(float)gridIdx/grid->nGridX(), (float)gridIdy/grid->nGridY(), (float)gridIdz/grid->nGridZ())
						);

			gridIdx = (gridIdx + 1) % grid->nGridX();
			if(gridIdx == 0)
				gridIdy = (gridIdy + 1) % grid->nGridY();
			if(gridIdx == 0 && gridIdy == 0)
				gridIdz = (gridIdz + 1) % grid->nGridZ();
	}

	//this->addRenderable(boundingBox);
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

