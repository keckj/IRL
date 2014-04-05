#include "sidePanel.hpp"
#include "sidePanel.moc"

#include "qtgui/globalGuiVars.hpp"
#include "image/image.hpp"

#include <iostream>


SidePanel::SidePanel(QWidget *parent) : QWidget(parent), sliceImage(0), axeChanged(false) {

		qtgui::sidepanel::sidepanel = this;
	
		//colortable for grayscale img
		for (int i = 0; i < 255; i++) {
			grayScaleColorTable.push_back(qRgb(i,i,i));
		}
	
		this->setStyleSheet("QWidget {background-color: white;}");
		this->setAutoFillBackground(true);
		
		//GroupBox
		renderGroupBox = new QGroupBox("Render property");
		sliceGroupBox = new QGroupBox("Slices");
		
		renderGroupBox->setStyleSheet(
										"QGroupBox {"
										"    border: 1px solid gray;"
										"    border-radius: 9px;"
										"    margin-top: 0.5em;"
										"}"
										""
										"QGroupBox::title {"
										"    subcontrol-origin: margin;"
										"    left: 20px;"
										"    padding: 0 3px 0 3px;"
										"}");

		sliceGroupBox->setStyleSheet(
										"QGroupBox {"
										"    border: 1px solid gray;"
										"    border-radius: 9px;"
										"    margin-top: 0.5em;"
										"}"
										""
										"QGroupBox::title {"
										"    subcontrol-origin: margin;"
										"    left: 20px;"
										"    padding: 0 3px 0 3px;"
										"}");

		//Labels
		thresholdLabel = new QLabel();
		currentSliceLabel = new QLabel();
		axeLabel = new QLabel("Axe :");
		sliceLabel = new QLabel();
		sliceLabel->setMinimumHeight(400);
		sliceLabel->setMinimumWidth(200);
		sliceLabel->setScaledContents(true);
		sliceLabel->setFixedSize(0,0);

		//CheckBox
		realTimeRenderCheckBox = new QCheckBox("Real time voxel rendering");
		realTimeRenderCheckBox->setChecked(qtgui::viewer::drawViewerBool);
		connect(realTimeRenderCheckBox, SIGNAL(stateChanged(int)), this, SLOT(realTimeRenderUpdate(int)));


		//PushButton
		renderPushButton = new QPushButton("Render");
		renderPushButton->setMaximumWidth(200);
		connect(renderPushButton, SIGNAL(clicked()), this, SLOT(renderOneTime()));
	
		//ComboBox
		axeComboBox = new QComboBox();
		axeComboBox->setMaximumWidth(100);
		axeComboBox->addItem("Axe X");
		axeComboBox->addItem("Axe Y");
		axeComboBox->addItem("Axe Z");
		connect(axeComboBox, SIGNAL(activated(int)), this, SLOT(axeUpdate(int)));
		
		//Slider
		thresholdSlider = new QSlider(Qt::Horizontal); 
		thresholdSlider->setMaximumWidth(400);
		thresholdSlider->setRange(0,255);
		thresholdSlider->setPageStep(10);
		thresholdSlider->setSingleStep(10);
		thresholdSlider->setTracking(true);
		connect(thresholdSlider, SIGNAL(valueChanged(int)), this, SLOT(thresholdSliderUpdate(int)));

		sliceSlider = new QSlider(Qt::Horizontal);
		sliceSlider->setMaximumWidth(400);
		sliceSlider->setRange(0,0);
		sliceSlider->setSingleStep(5);
		sliceSlider->setPageStep(5);
		sliceSlider->setTracking(true);
		sliceSlider->setValue(0);
		connect(sliceSlider, SIGNAL(valueChanged(int)), this, SLOT(sliceSliderUpdate(int)));

		//QPixmap
		slicePixMap = new QPixmap();
		
		//Layouts
		layout = new QBoxLayout(QBoxLayout::TopToBottom, this);
		renderLayout = new QBoxLayout(QBoxLayout::TopToBottom, renderGroupBox);
		sliceLayout = new QBoxLayout(QBoxLayout::TopToBottom, sliceGroupBox);
		
		renderGroupBox->setLayout(renderLayout);
		sliceGroupBox->setLayout(sliceLayout);
		this->setLayout(layout);
		

		//Include widgets
		renderLayout->setSpacing(20);
		renderLayout->addWidget(realTimeRenderCheckBox);
		renderLayout->addWidget(renderPushButton);
		renderLayout->addWidget(thresholdLabel);
		renderLayout->addWidget(thresholdSlider);
		
		sliceLayout->setSpacing(20);
		sliceLayout->addWidget(axeLabel);
		sliceLayout->addWidget(axeComboBox);
		sliceLayout->addWidget(currentSliceLabel);
		sliceLayout->addWidget(sliceSlider);
		sliceLayout->addWidget(sliceLabel);
		sliceLayout->setAlignment(sliceLabel, Qt::AlignHCenter);
		
		layout->setSpacing(50);
		layout->addWidget(renderGroupBox);
		layout->addWidget(sliceGroupBox);
		layout->addStretch();
		
		//signals to initialize
		updateThreshold();
		sliceSliderUpdate(0);
		axeUpdate(0);
		realTimeRenderUpdate(Qt::Checked);
		renderSlice();
}
		
SidePanel::~SidePanel() {
}
		
void SidePanel::thresholdSliderUpdate(int value) {
	if(value != qtgui::viewer::viewerThreshold) {
		qtgui::viewer::viewerThreshold = value;
		char str[100];
		sprintf(str, "Current voxel threshold : %i", value);
		thresholdSlider->setValue(value);
		thresholdLabel->setText(str);
		emit drawVoxels();
	}
}
		
void SidePanel::sliceSliderUpdate(int value) {
	qtgui::sidepanel::sliceId = value;
	char str[100];
	sprintf(str, "Current slice : %i", value);
	sliceSlider->setValue(value);
	currentSliceLabel->setText(str);
	renderSlice();
}

void SidePanel::axeUpdate(int axe_id) {
	switch(axe_id) {
		case 0:
			qtgui::sidepanel::sliceAxe = Image::SliceAxe::AXE_X;
			sliceSlider->setRange(0, qtgui::voxelGrid->width()-1);
			sliceSlider->setValue(0);
			break;
		case 1:
			qtgui::sidepanel::sliceAxe = Image::SliceAxe::AXE_Y;
			sliceSlider->setRange(0, qtgui::voxelGrid->height()-1);
			sliceSlider->setValue(0);
			break;
		case 2:
			qtgui::sidepanel::sliceAxe = Image::SliceAxe::AXE_Z;
			sliceSlider->setRange(0, qtgui::voxelGrid->length()-1);
			sliceSlider->setValue(0);
			break;
		default:
			assert(false);
	}

	renderSlice();
}
void SidePanel::realTimeRenderUpdate(int state) {
	qtgui::viewer::drawViewerBool = (state == Qt::Checked);
	renderPushButton->setEnabled(!(state == Qt::Checked));
	emit drawVoxels();
}
void SidePanel::renderOneTime() {
	qtgui::viewer::drawOneTime = true;
	emit drawVoxels();
}
		
void SidePanel::updateThreshold() {
	char str[100];
	sprintf(str, "Current voxel threshold : %i", qtgui::viewer::viewerThreshold);
	thresholdSlider->setValue(qtgui::viewer::viewerThreshold);
	thresholdLabel->setText(str);
}

void SidePanel::renderSlice() {
	
	if(sliceImage) {
		delete sliceImage;
	}
	
	sliceImage = Image::generateParallelSlice(
				qtgui::sidepanel::sliceId, 
				*qtgui::voxelGrid, 
				qtgui::sidepanel::sliceAxe);
	sliceImage->setColorCount(256);
	sliceImage->setColorTable(grayScaleColorTable);
	
	slicePixMap->convertFromImage(*sliceImage);
	resizeEvent(0);
}

void SidePanel::resizeEvent(QResizeEvent *event) {

	unsigned int maxHeight = 0.6*this->height();
	unsigned int maxWidth = 0.9*sliceGroupBox->width();
	
	float alpha_w = (float) (maxWidth)/slicePixMap->width();
	float alpha_h = (float) (maxHeight)/slicePixMap->height();
	
	QPixmap pix;

	if(alpha_w < alpha_h) {
		pix = slicePixMap->scaledToWidth(maxWidth);
	}
	else {
		pix = slicePixMap->scaledToHeight(maxHeight);
	}

	sliceLabel->setPixmap(pix);
	sliceLabel->setFixedSize(pix.size());
}

