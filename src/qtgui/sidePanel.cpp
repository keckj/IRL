#include "sidePanel.hpp"

SidePanel::SidePanel(QWidget *parent) : QWidget(parent) {
	
		this->setStyleSheet("QWidget {background-color: red;}");
		this->setAutoFillBackground(true);
}
