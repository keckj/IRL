
TARGET = main
CONFIG *= qt opengl release
QT *= opengl xml

HEADERS = *.h
SOURCES = *.cpp

# Linux
INCLUDEPATH *= /usr/include
LIBS *= -L/usr/lib -lQGLViewer -lGLU -lglut

