#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui>
#include <QGLWidget>
#include <QGLViewer/qglviewer.h>
#include "grid/voxelGrid.hpp"

#include "qtgui/menuBar.hpp"
#include "qtgui/statusBar.hpp"

class MainWindow : public QMainWindow {
	
	public:
		MainWindow(const VoxelGrid *v, 
				bool drawVoxels = true, unsigned char viewerThreshold = 127);
		~MainWindow();

	private:
		const VoxelGrid *voxelGrid;

		QWidget *central, *test;
		QGLViewer *viewer;
		QSplitter *layout;
		MenuBar *menuBar;
		StatusBar *statusBar;

		bool drawVoxels;
		unsigned char viewerThreshold;

		void keyPressEvent(QKeyEvent *k);
};

#endif /* end of include guard: MAINWINDOW_H */

