#include "qtgui/mainWindow.hpp"
#include <QKeyEvent>
#include <QStatusBar>

MainWindow::MainWindow(
		const VoxelGrid *voxelGrid,
		bool drawVoxels, 
		unsigned char viewerThreshold) :
	QMainWindow(0), 
	voxelGrid(voxelGrid),
	drawVoxels(drawVoxels), 
	viewerThreshold(viewerThreshold) {
	
		QDesktopWidget widget;
		QRect mainScreenSize = widget.availableGeometry(widget.primaryScreen());

		menuBar = new MenuBar(this);
		this->setMenuBar(menuBar);

		statusBar = new StatusBar(this);
		this->setStatusBar(statusBar);


		this->setWindowTitle("Poulpy");
		this->resize(mainScreenSize.width(),mainScreenSize.height());
		this->setStyleSheet("QMainWindow { background-color: white; }");
		this->setAutoFillBackground(true);


		layout = new QSplitter(Qt::Horizontal, this);
		
		test = new QWidget(layout);
		test->resize(100,600);
		test->setStyleSheet("QWidget {background-color: red;}");
		test->setAutoFillBackground(true);

		viewer = new QGLViewer(layout);

		layout->addWidget(viewer);
		layout->addWidget(test);
		layout->setStretchFactor(0,2);
		layout->setStretchFactor(1,5);

		setCentralWidget(layout);

		show();
	}

MainWindow::~MainWindow() {
	delete test;
	delete viewer;
	delete central;
	delete layout;
}

void MainWindow::keyPressEvent(QKeyEvent *k) {
	
	switch(k->key()) {
		case Qt::Key_Escape:
			this->close();
			break;
		
	}
}
