#include <QKeyEvent>
#include <QStatusBar>

#include "qtgui/mainWindow.hpp"
#include "qtgui/mainWindow.moc"
#include "qtgui/globalGuiVars.hpp"


MainWindow::MainWindow() :
slider(0), voxelViewer(0), sidePanel(0), menuBar(0), statusBar(0)
{
		qtgui::mainWindow = this;
	
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

		connect(voxelViewer, SIGNAL(updateThreshold()), sidePanel, SLOT(updateThreshold()));
		connect(sidePanel, SIGNAL(drawVoxels()), voxelViewer, SLOT(drawVoxels()));
		
		connect(voxelViewer, SIGNAL(childKeyEvent(QKeyEvent *)), this, SLOT(childKeyEvent(QKeyEvent *)));

		slider->addWidget(voxelViewer);
		slider->addWidget(sidePanel);
		slider->setStretchFactor(0,6);
		slider->setStretchFactor(1,3);

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
		
void MainWindow::childKeyEvent(QKeyEvent *k) {
	keyPressEvent(k);
}
