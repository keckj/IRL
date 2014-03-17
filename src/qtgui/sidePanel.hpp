#ifndef SIDEPANEL_H
#define SIDEPANEL_H

#include <QBoxLayout>
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QComboBox>
#include <QSlider>
#include <QGroupBox>

class SidePanel : public QWidget {
	Q_OBJECT

	public:
		SidePanel(QWidget *parent = 0);
		~SidePanel();
	
	signals:
		void drawVoxels();

	public slots:
		void updateThreshold();

	private slots:
		void thresholdSliderUpdate(int value);
		void sliceSliderUpdate(int value);
		void axeUpdate(int axe_id);
		void realTimeRenderUpdate(int state);
		void renderOneTime();
	
	private:
		QBoxLayout *layout, *renderLayout, *sliceLayout;
		QGroupBox *renderGroupBox, *sliceGroupBox;
		QLabel *thresholdLabel, *axeLabel, *currentSliceLabel;
		QSlider *thresholdSlider, *sliceSlider;
		QPushButton *renderPushButton;
		QCheckBox *realTimeRenderCheckBox;
		QComboBox *axeComboBox;

};


#endif /* end of include guard: SIDEPANEL_H */
