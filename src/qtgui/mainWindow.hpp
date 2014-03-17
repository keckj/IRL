#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui>
#include <QGLWidget>

#include "qtgui/menuBar.hpp"
#include "qtgui/statusBar.hpp"
#include "qtgui/sidePanel.hpp"
#include "qtgui/voxelViewer.hpp"

class MainWindow : public QMainWindow {
	Q_OBJECT
	
	public:
		MainWindow();
		~MainWindow();

	private:
		QSplitter *slider;
		VoxelViewer *voxelViewer;
		SidePanel *sidePanel;
		MenuBar *menuBar;
		StatusBar *statusBar;

		void keyPressEvent(QKeyEvent *k);
	
	private slots:
		void childKeyEvent(QKeyEvent *k);
};

#endif /* end of include guard: MAINWINDOW_H */

