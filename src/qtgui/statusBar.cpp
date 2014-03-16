#include "statusBar.hpp"

StatusBar::StatusBar(QWidget *parent) : QStatusBar(parent) {
	this->showMessage("Nothing to be done !");
}
