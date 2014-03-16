#include <QKeyEvent>
#include <QStatusBar>

#include "qtgui/mainWindow.hpp"
#include "qtgui/globalGuiVars.hpp"

MainWindow::MainWindow() :
slider(0), voxelViewer(0), sidePanel(0), menuBar(0), statusBar(0)
{
	
		QDesktopWidget widget;
		QRect mainScreenSize = widget.availableGeometry(widget.primaryScreen());

		this->setWindowTitle("Voxel Engine v1.2");
		this->resize(mainScreenSize.width(),mainScreenSize.height());
		this->setStyleSheet("QMainWindow { background-color: white; }");
		this->setAutoFillBackground(true);
		
		menuBar = new MenuBar(this);
		this->setMenuBar(menuBar);

		statusBar = new StatusBar(this);
		this->setStatusBar(statusBar);

		slider = new QSplitter(Qt::Horizontal, this);
		
		voxelViewer = new VoxelViewer(0.001, slider);
		sidePanel = new SidePanel(slider);

		slider->addWidget(voxelViewer);
		slider->addWidget(sidePanel);
		slider->setStretchFactor(0,2);
		slider->setStretchFactor(1,5);

		this->setCentralWidget(slider);
		this->show();
	}

MainWindow::~MainWindow() {
}

void MainWindow::keyPressEvent(QKeyEvent *k) {
	
	switch(k->key()) {
		case Qt::Key_Escape:
			this->close();
			break;
		
	}
}
