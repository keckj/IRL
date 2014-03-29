#pragma once
#include "grid/voxelGrid.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <list>
#include <QImage>

using namespace std;
using namespace cv;

class Image {

	private:
		list<Mat*> images;
		
	public:
		int loadImageFolder(const char *location);
		int loadImage(const char *name);

		void loadLocalizedUSImages(
				string const & folderName, 
				int *nImage, int *imgWidth, int *imgHeight, float *deltaX, float *deltaY,  
				float ***offsets, float ***rotations, float **data,
				bool pageLockedMemory
				);

		void computeImageFiltering();
		void computeGradientVectorFlow();
		
		bool compareDataOrder(string const & str1, string const & str2);
		
		static void displayImage(Mat &m);
		static void displayQTImage(Mat &m);

		static void filter1D(float *data, int nData, int size, float sigma);
	
		enum SliceAxe {AXE_X, AXE_Y, AXE_Z};
		static QImage *generateParallelSlice(unsigned int nSlice,
							VoxelGrid<unsigned char,PinnedCPUResource,GPUResource> &grid, SliceAxe axe);
};
