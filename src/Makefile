
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
EXCLUDED_SUBDIRS = . .. old " "

SRCDIR = $(realpath .)
OBJDIR = .

TARGET = main

EXT = c C cc cpp s S asm cu
SRC = $(notdir $(wildcard $(EXT)))
OBJ = $(addprefix $(OBJDIR), $(addsuffix .o, $(basename $(SRC)))))

# Règles
all: $(TARGET)

debug: CFLAGS += $(DEBUGFLAGS)
debug: CXXFLAGS += $(DEBUGFLAGS) 
debug: NVCCFLAGS += $(DEBUGFLAGS) 
debug: $(TARGET)

profile: CFLAGS += $(PROFILINGFLAGS)
profile: CXXFLAGS += $(PROFILINGFLAGS)
profile: NVCCFLAGS += $(PROFILINGFLAGS)
profile: $(TARGET)

release: CFLAGS += $(RELEASEFLAGS)
release: CXXFLAGS += $(RELEASEFLAGS)
release: NVCCFLAGS += $(RELEASEFLAGS)
release: $(TARGET)

$(TARGET): 
	@echo
	@echo "Actual dir is " $(SRCDIR)
	$(eval SUBDIRS = $(shell find $(SRCDIR)/* -maxdepth 0 -type d))
	@echo "Subdirs are " $(SUBDIRS)
	
	$(words SUBDIRS)
	$(foreach DIR, $(SUBDIRS), $(MAKE) all SRCDIR=$(DIR) )
	@echo


# "-" pour enlever les messages d'erreurs
# "@" pour silent


.PHONY: clean cleanall

clean:
	-@rm -rf $(OBJDIR)*.o

cleanall: clean
	-@rm -f $(TARGET) $(TARGET).out 
