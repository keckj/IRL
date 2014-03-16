#include "sidePanel.hpp"
#include "sidePanel.moc"

#include <iostream>


SidePanel::SidePanel(QWidget *parent) : QWidget(parent) {
	
		this->setStyleSheet("QWidget {background-color: white;}");
		this->setAutoFillBackground(true);

		QBoxLayout *layout = new QBoxLayout(QBoxLayout::TopToBottom, this);
		this->setLayout(layout);


		QSlider *slider = new QSlider(Qt::Horizontal, this); 
		slider->setMinimum(0);
		slider->setMaximum(255);
		slider->setPageStep(10);
		slider->setSingleStep(10);
		connect(slider, SIGNAL(valueChanged(int)), this, SLOT(sliderUpdate(int)));

		layout->addWidget(slider);
}
		
SidePanel::~SidePanel() {
}
		
void SidePanel::sliderUpdate(int value) {
	std::cout << value << std::endl;
}
