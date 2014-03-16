#include "statusBar.hpp"


StatusBar::StatusBar(QWidget *parent) : 
QStatusBar(parent),
progressBar(0), message(0)
{
	progressBar = new QProgressBar(this);
	progressBar->setMinimum(0);
	progressBar->setMaximum(100);
	progressBar->setValue(100);
	progressBar->setMaximumSize(200,20);

	message = new QLabel(this);
	message->setText("Nothing to be done !");
	message->setMaximumSize(300,20);

	this->addWidget(message,1);
	this->addWidget(progressBar,1);
	
	
}
	
StatusBar::~StatusBar() {
}
