#ifndef STATUSBAR_H
#define STATUSBAR_H

#include <QWidget>
#include <QStatusBar>
#include <QProgressBar>
#include <QHBoxLayout>
#include <QLabel>

class StatusBar : public QStatusBar {
	
	public:
		StatusBar(QWidget *parent = 0);
		~StatusBar();

	private:
		QProgressBar *progressBar;
		QLabel *message;
};


#endif /* end of include guard: STATUSBAR_H */
