
#ifndef VOXELVIEWER_H
#define VOXELVIEWER_H

#define _CUDA_VIEWER

#include <QWidget>
#include "viewer/viewer.h"
#include "viewer/voxelRenderer.hpp"
#include "viewer/boundingBox.hpp"

class VoxelViewer : public Viewer {
	Q_OBJECT

	public:
		VoxelViewer(float alpha, QWidget *parent = 0);
		~VoxelViewer();
		
		void keyPressEvent(QKeyEvent*);
	
	signals:
		void updateThreshold();
		void childKeyEvent(QKeyEvent *keyEvent);

	public slots:
		void drawVoxels();

	private:

		float alpha;
		VoxelRenderer *voxelRenderer;
		BoundingBox *boundingBox;



};


#endif /* end of include guard: VOXELVIEWER_H */
