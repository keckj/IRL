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
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <QRgb>

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
		QLabel *thresholdLabel, *axeLabel, *currentSliceLabel, *sliceLabel;
		QSlider *thresholdSlider, *sliceSlider;
		QPushButton *renderPushButton;
		QCheckBox *realTimeRenderCheckBox;
		QComboBox *axeComboBox;
		QPixmap *slicePixMap;
		QImage *sliceImage;

		QVector<QRgb> grayScaleColorTable;

		bool axeChanged;

		void renderSlice();

		void resizeEvent(QResizeEvent *event);

};


#endif /* end of include guard: SIDEPANEL_H */
