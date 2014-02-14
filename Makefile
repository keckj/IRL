
####################
### LIB EXTERNES ###
####################
OPENCV_LIBPATH = -L/usr/lib
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui

CUDA_INCLUDEPATH = -I/usr/local/cuda-5.5/include
CUDA_LIBPATH = -L/usr/local/cuda-5.5/lib
CUDA_LIBS = -lcuda

OPENCL_INCLUDEPATH = -I/opt/AMDAPP/include
OPENCL_LIBPATH = -L/opt/AMDAPP/lib/x86
OPENCL_LIBS = -lOpenCL

OPENGL_LIBPATH = -L/usr/X11R6/lib64
OPENGL_INCLUDEPATH = -I/usr/lib64/qt4/mkspecs/linux-g++-64 -I/usr/include/QtCore -I/usr/include/QtGui -I/usr/include/QtOpenGL -I/usr/include/QtXml -I/usr/include -I/usr/X11R6/include 
OPENGL_LIBS = -lglut -lQGLViewer -lGLU -lGL -lQtXml -lQtOpenGL -lQtGui -lQtCore -lpthread 
#OPENGL_LIBS = -lglfw3 -lGL -lGLEW -lGLU -lX11 -lXxf86vm -lXrandr -lpthread -lXi

####################

#Compilateurs
LINK= nvcc
LINKFLAGS= -O3 -arch=sm_20 -Xcompiler -Wextra -m64
LDFLAGS= $(OPENGL_LIBS) $(OPENCV_LIBS) $(CUDA_LIBS) -llog4cpp

INCLUDE = -I$(SRCDIR) $(OPENGL_INCLUDEPATH) $(CUDA_INCLUDEPATH) $(OPENCV_INCLUDEPATH)
LIBS = $(OPENCL_LIBPATH) $(CUDA_LIBPATH) $(OPENCV_LIBPATH)

CC=gcc
CFLAGS= -W -Wall -Wextra -pedantic -std=c99

CXX=g++
CXXFLAGS= -W -Wall -Wextra -pedantic -std=c++11

AS = nasm
ASFLAGS= -f elf64

# Autres flags 
DEBUGFLAGS= -g -O0
PROFILINGFLAGS= -pg
RELEASEFLAGS= -O3

# Source et destination des fichiers
TARGET = main

SRCDIR = $(realpath .)/src
OBJDIR = $(realpath .)/obj
TEST= old viewer
EXCLUDED_SUBDIRS = $(foreach DIR, $(TEST), $(call subdirs, $(SRCDIR)/$(DIR)))
SUBDIRS =  $(filter-out $(EXCLUDED_SUBDIRS), $(call subdirs, $(SRCDIR)))

SRC_EXTENSIONS = c C cc cpp s S asm cu
WEXT = $(addprefix *., $(SRC_EXTENSIONS))
SRC = $(foreach DIR, $(SUBDIRS), $(foreach EXT, $(WEXT), $(wildcard $(DIR)/$(EXT))))
OBJ = $(subst $(SRCDIR), $(OBJDIR), $(addsuffix .o, $(basename $(SRC))))

include rules.mk
