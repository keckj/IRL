#ifndef SIDEPANEL_H
#define SIDEPANEL_H

#include <QWidget>
#include <QSlider>
#include <QBoxLayout>

class SidePanel : public QWidget {
	Q_OBJECT

	public:
		SidePanel(QWidget *parent = 0);
		~SidePanel();
	private slots:
		void sliderUpdate(int value);

};


#endif /* end of include guard: SIDEPANEL_H */
