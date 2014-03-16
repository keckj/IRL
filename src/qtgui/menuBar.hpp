
#ifndef MENUBAR_H
#define MENUBAR_H

#include <QMenu>
#include <QWidget>
#include <QMenuBar>

class MenuBar : public QMenuBar {
	
	public:
		MenuBar(QWidget *parent = 0);

	private:
		QAction *quit;
		QMenu *file;
};

#endif /* end of include guard: MENUBAR_H */
