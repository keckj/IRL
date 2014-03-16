#include "menuBar.hpp"
#include <QtGui>

MenuBar::MenuBar(QWidget *parent) : QMenuBar(parent) {
		file = this->addMenu("&File");
		quit = new QAction("&Quit", this);
		
		file->addAction(quit);
		connect(quit, SIGNAL(triggered()), qApp, SLOT(quit()));
}
