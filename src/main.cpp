#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "cuda.h"
#include "cuda_runtime.h"

#include "utils/log.hpp"
#include "utils/cudaUtils.hpp"
#include "image/image.hpp"
#include "image/LocalizedUSImage.hpp"

using namespace std;
using namespace cv;

extern void testKernel(const int nImages, const int imgWidth, const int imgHeight, float *float_data, unsigned char *char_data);

int main( int argc, const char* argv[] )
{
	initLogs();

	Image im;
	int nImages;
	float *x,*y,*z, **R, **data;
	float dx, dy;
	int w, h;
	//im.loadLocalizedUSImages("data/imagesUS/", &nImages, &w, &h, &dx, &dy, &x, &y, &z, &R);
	im.loadLocalizedUSImages("data/processedImages/", &nImages, &w, &h, &dx, &dy, &x, &y, &z, &R, &data);
	
	
	const int imgWidth = w;
	const int imgHeight = h;
	const float deltaGrid = 0.1;
	const float deltaX = dx;
	const float deltaY = dy;
	const float imgRealWidth = w*dx;
	const float imgRealHeight = h*dy;
	const unsigned int imageSize = imgWidth * imgHeight * nImages;
	
	log_console.infoStream() << "\tImage number : " << nImages;
	log_console.infoStream() << "\tImages size : " << imgWidth << " x " << imgHeight << " (px)";
	log_console.infoStream() << "\tSensor precision : " << deltaX*1000 << " x " << deltaY*1000 << " (µm)";
	log_console.infoStream() << "\tImages real size : " << imgRealWidth << " x " << imgRealHeight << " (mm)";

	float posX, posY, posZ;
	float xMin, xMax, yMin, yMax, zMin, zMax;

	float inf= std::numeric_limits<float>::infinity();
	xMax = -inf; yMax = -inf; zMax = -inf;	
	xMin = +inf; yMin = +inf; zMin = +inf;	
	
	log_console.infoStream() << "Computing bounding box...";

	for (int i = 0; i < nImages; i++) {
		posX = x[i]; posY = y[i]; posZ = z[i];

		xMin = (posX < xMin ? posX : xMin);
		yMin = (posY < yMin ? posY : yMin);
		zMin = (posZ < zMin ? posZ : zMin);
		xMax = (posX > xMax ? posX : xMax);
		yMax = (posY > yMax ? posY : yMax);
		zMax = (posZ > zMax ? posZ : zMax);
		
		posX = R[0][0]*imgRealWidth + R[0][1]*imgRealHeight + R[0][2]*0.0f + posX;
		posY = R[1][0]*imgRealWidth + R[1][1]*imgRealHeight + R[1][2]*0.0f + posY;
		posZ = R[2][0]*imgRealWidth + R[2][1]*imgRealHeight + R[2][2]*0.0f + posZ;

		xMin = (posX < xMin ? posX : xMin);
		yMin = (posY < yMin ? posY : yMin);
		zMin = (posZ < zMin ? posZ : zMin);
		xMax = (posX > xMax ? posX : xMax);
		yMax = (posY > yMax ? posY : yMax);
		zMax = (posZ > zMax ? posZ : zMax);
	}

	const float boxWidth = xMax - xMin;
	const float boxHeight = yMax - yMin;
	const float boxLentgh = zMax - zMin;
	const unsigned int minVoxelGridWidth = ceil(boxWidth/deltaGrid);
	const unsigned int minVoxelGridHeight = ceil(boxHeight/deltaGrid);
	const unsigned int minVoxelGridLength = ceil(boxLentgh/deltaGrid);
	const unsigned int voxelGridWidth = pow(2, ceil(log2(boxWidth/deltaGrid)));
	const unsigned int voxelGridHeight = pow(2, ceil(log2(boxHeight/deltaGrid)));
	const unsigned int voxelGridLength = pow(2, ceil(log2(boxLentgh/deltaGrid)));
	const unsigned int gridSize = voxelGridWidth * voxelGridHeight * voxelGridLength;
	
	log_console.infoStream() << "\tpMin = (" << xMin << "," << yMin << "," << zMin << ")";
	log_console.infoStream() << "\tpMax = (" << xMax << "," << yMax << "," << zMax << ")";
	log_console.infoStream() << "\tBox Size : " << boxWidth << "x" << boxHeight << "x" << boxLentgh << " (mm)";

	log_console.infoStream() << "Voxel grid precision set to " << deltaGrid*1000 << " µm";
	log_console.infoStream() << "Minimum grid size : " 
		<< minVoxelGridWidth << "x" << minVoxelGridHeight << "x" << minVoxelGridLength;
	log_console.infoStream() << "Effective grid size : " 
		<< voxelGridWidth << "x" << voxelGridHeight << "x" << voxelGridLength;
	log_console.infoStream() << "Effective grid memory size (unsigned char) : " 
		<< gridSize/(1024*1024) * sizeof(unsigned char) << " MB";
	log_console.infoStream() << "Effective images memory size (float) : " 
		<< imageSize/(1024*1024) * sizeof(float) << " MB";
	log_console.infoStream() << "Effective images memory size (unsigned char) : " 
		<< imageSize/(1024*1024) * sizeof(unsigned char) << " MB";
	log_console.infoStream() << "The programm will need to use at least " 
		<< max((gridSize + imageSize)*sizeof(unsigned char), (sizeof(float)+sizeof(unsigned char))*imageSize)/(1024*1024)
		<< " MB of VRAM.";
	
	if(max((gridSize + imageSize)*sizeof(unsigned char), (sizeof(float)+sizeof(unsigned char))*imageSize)/(1024*1024) >= 1024) {
		log_console.warnStream() << "The programm will use more then 1GB of VRAM, please check if your GPU has enough memory !";
	}

	CudaUtils::logCudaDevices(log_console);

	float *device_float_data;
	unsigned char *device_char_data;
	unsigned char *host_char_data;

	host_char_data = (unsigned char *) malloc(nImages*imgWidth*imgHeight*sizeof(unsigned char));

	CHECK_CUDA_ERRORS(cudaSetDevice(0));	

	//float data
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &device_float_data, nImages*imgWidth*imgHeight*sizeof(float)));
	for (int i = 0; i < nImages; i++) {
		CHECK_CUDA_ERRORS(cudaMemcpy(device_float_data + i*imgWidth*imgHeight, data[i], imgWidth*imgHeight*sizeof(float), cudaMemcpyHostToDevice));
	}

	//unsigned char data
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &device_char_data, nImages*imgWidth*imgHeight*sizeof(unsigned char)));

	//call kernel
	testKernel(nImages, imgWidth, imgHeight, device_float_data, device_char_data);
	
	//copy back array
	CHECK_CUDA_ERRORS(cudaMemcpy(host_char_data, device_char_data, nImages*imgWidth*imgHeight, cudaMemcpyDeviceToHost));

	Mat m0(imgHeight, imgWidth, CV_32F, data[0]);
	m0.convertTo(m0, CV_8UC1);
	Mat m1(imgHeight, imgWidth, CV_8UC1, host_char_data);
	Image::displayImage(m0);
	Image::displayImage(m1);
	


	//free
	CHECK_CUDA_ERRORS(cudaFree(device_float_data));
	CHECK_CUDA_ERRORS(cudaFree(device_char_data));
	free(host_char_data);


	return EXIT_SUCCESS;


	LocalizedUSImage::initialize();
	LocalizedUSImage img("data/processedImages/" , "IQ[data #123 (RF Grid).mhd");

	Mat m(img.getHeight(), img.getWidth(), CV_64F, img.getImageData());

	double min, max;
	minMaxIdx(m, &min, &max);
	cout << "\nvals \t" << min << "\t" << max << endl;
	Mat hist;
	int hist_size = 128;
	float range[] = {(float) min, (float) max};
	const float *hist_range = {range};
	calcHist(&m, 1, 0, Mat(), hist, 1, &hist_size, &hist_range, true, false);

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/hist_size );
	Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	for( int i = 1; i < hist_size; i++ )
	{
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
				Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
				Scalar( 255, 0, 0), 2, 8, 0  );
	}

	/// Display
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	imshow("calcHist Demo", histImage );

	//cout << img;
	Mat m2,m3;
	m.convertTo(m2, CV_8UC1);
	m.convertTo(m3, CV_8UC1);
	GaussianBlur(m2, m2, Size(9,9), 5.0);
	int lowThreshold = 30;
	int kernel_size = 3;
	Canny(m2, m2, lowThreshold, lowThreshold*3, kernel_size);
	m2.copyTo(m3, m2);
	Image::displayImage(m3);

	log_console.info("END OF MAIN PROGRAMM");

	return EXIT_SUCCESS;
}

