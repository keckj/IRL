
OPENCV_LIBPATH = -L/usr/lib
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui

CUDA_INCLUDEPATH = -I/usr/local/cuda-5.5/include
CUDA_LIBPATH = -L/usr/local/cuda-5.5/lib
CUDA_LIBS = -lcuda

OPENCL_INCLUDEPATH = -I/opt/AMDAPP/include
OPENCL_LIBPATH = -L/opt/AMDAPP/lib/x86
OPENCL_LIBS = -lOpenCL

OPENGL_INGLUDEPATH =
OPENGL_LIBPATH =
OPENGL_LIBS = -lglfw3 -lGL -lGLEW -lGLU -lX11 -lXxf86vm -lXrandr -lpthread -lXi
#OPENGL_LIBS = -lGL -lGLEW -lglfw3 -lX11 -lsfml-graphics -lsfml-window -lsfml-system
###############################

#macros
containing = $(foreach v,$2,$(if $(findstring $1,$v),$v))
not_containing = $(foreach v,$2,$(if $(findstring $1,$v),,$v))
subdirs = $(shell find $1 -type d)

#LDFLAGS= -lSDL2main -lSDL2 -lSDL2_ttf -lSDL2_mixer -lGL -lGLU
LDFLAGS= $(OPENGL_LIBS) $(OPENCV_LIBS) -llog4cpp $(CUDA_LIBS)#$(OPENCL_LIBS) $(OPENCV_LIBS) 
INCLUDE = $(OPENGL_INCLUDEPATH) $(CUDA_INCLUDEPATH)# $(OPENCL_INCLUDEPATH) $(OPENCV_INCLUDEPATH) #
LIBS = $(OPENCL_LIBPATH) $(CUDA_LIBPATH)#$lOpenCL $(OPENGL_LIBPATH) $(OPENCV_LIBPATH) #$(CUDA_LIBPATH)

#Compilateurs
CC=gcc
CFLAGS= -W -Wall -Wextra -pedantic -std=c99

CXX=g++
CXXFLAGS= -W -Wall -Wextra -pedantic -std=c++11

AS = nasm
ASFLAGS= -f elf64

NVCC=nvcc
NVCCFLAGS= -O3 -arch=sm_20 -Xcompiler -Wextra -m64

# Autres règles
DEBUGFLAGS= -g -O0
PROFILINGFLAGS= -pg
RELEASEFLAGS= -O3

# Source et destination des fichiers
TARGET = main

SRCDIR = $(realpath src)
OBJDIR = $(realpath obj)
EXCLUDED_SUBDIRS = $(call subdirs, src/old)
SUBDIRS =  $(filter-out $(EXCLUDED_SUBDIRS), $(call subdirs, $(SRCDIR)))

EXTENSIONS = $(addprefix *., c C cc cpp s S asm cu)
SRC = $(foreach DIR, $(SUBDIRS), $(foreach EXT, $(EXTENSIONS), $(wildcard $(DIR)/$(EXT))))
OBJ = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))

# Règles
all: $(TARGET)

debug: 
	$(eval CFLAGS += $(DEBUGFLAGS))
	$(eval CXXFLAGS += $(DEBUGFLAGS))
	$(eval NVCCFLAGS += $(DEBUGFLAGS))
debug: all

profile: 
	$(eval CFLAGS += $(PROFILINGFLAGS))
	$(eval CXXFLAGS += $(PROFILINGFLAGS))
	$(eval NVCCFLAGS += $(PROFILINGFLAGS))
	$(eval $(TARGET))

release: CFLAGS += $(RELEASEFLAGS)
release: CXXFLAGS += $(RELEASEFLAGS)
release: NVCCFLAGS += $(RELEASEFLAGS)
release: $(TARGET)

$(TARGET): 
	@echo $(SRCDIR)
	@echo
	@echo $(EXCLUDED_SUBDIRS)
	@echo
	@echo $(SUBDIRS)
	@echo
	@echo $(SRC)
	@echo
	@echo $(OBJ)

# "-" pour enlever les messages d'erreurs
# "@" pour silent


.PHONY: clean cleanall

clean:
	-@rm -rf $(OBJDIR)*.o

cleanall: clean
	-@rm -f $(TARGET) $(TARGET).out 
