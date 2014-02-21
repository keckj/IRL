#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "cuda.h"
#include "cuda_runtime.h"

#include <QApplication>
#include <QWidget>
#include "utils/log.hpp"
#include "utils/cudaUtils.hpp"
#include "image/image.hpp"
#include "image/LocalizedUSImage.hpp"
#include "viewer/viewer.h"
#include "viewer/voxelRenderer.hpp"
#include "kernels/kernels.hpp"


using namespace std;
using namespace cv;
using namespace kernel;

int main( int argc, char** argv)
{
	initLogs();


	//default values
	float dG = 0.5f;
	int thres = 128;
	string dataSource("data/imagesUS/");
	

	//parsing input
	if(argc >= 2) { 
		if (argv[1][0] == '0')
			;
		else if (argv[1][0] == '1')
			dataSource = string("data/processedImages/");
		else if(argv[1][0] == '2')  
			dataSource = string("data/femur/");
		else {
			log_console.critStream() << "Failed to parse input argument 1 : " << argv[1] << " !";
			exit(1);
		}
	}
	if(argc >= 3) { 
			dG = atof(argv[2]);
	}
	if(argc >= 4) { 
			thres = atoi(argv[3]);
			if(thres < 0 || thres > 255) {
				log_console.critStream() << "Threshold must be set between 0 and 255 !";
				exit(1);
			}
	}
	if(argc >= 5) {
			log_console.critStream() << "Too much input arguments " << argv[3] << " !";
			exit(1);
	}
		
	
	//Loading data 
	Image im;
	int nImages;
	float *x,*y,*z, **R, **data;
	float dx, dy;
	int w, h;
	im.loadLocalizedUSImages(dataSource, &nImages, &w, &h, &dx, &dy, &x, &y, &z, &R, &data);

	const unsigned char viewerThreshold = (unsigned char) thres;
	const int imgWidth = w;
	const int imgHeight = h;
	const float deltaX = dx;
	const float deltaY = dy;
	const float deltaGrid = dG;
	const float imgRealWidth = imgWidth*dx;
	const float imgRealHeight = imgHeight*dy;
	const unsigned long imageSize = imgWidth * imgHeight * nImages;

	log_console.infoStream() << "\tImage number : " << nImages;
	log_console.infoStream() << "\tImages size : " << imgWidth << " x " << imgHeight << " (px)";
	log_console.infoStream() << "\tSensor precision : " << deltaX*1000 << " x " << deltaY*1000 << " (µm)";
	log_console.infoStream() << "\tImages real size : " << imgRealWidth << " x " << imgRealHeight << " (mm)";

	//filtering positions
	//Image::filter1D(x, nImages, 5, 1.0f);	
	//Image::filter1D(y, nImages, 5, 1.0f);	
	//Image::filter1D(z, nImages, 5, 1.0f);	


	//compute bounding box
	float posX, posY, posZ;
	float xMin, xMax, yMin, yMax, zMin, zMax;

	float inf= std::numeric_limits<float>::infinity();
	xMax = -inf; yMax = -inf; zMax = -inf;	
	xMin = +inf; yMin = +inf; zMin = +inf;	

	log_console.infoStream() << "Computing bounding box...";

	float vects[4][3] = {{0.0f,0.0f,0.0f}, {imgRealWidth,0.0f,0.0f}, {0.0f,imgRealHeight,0.0f}, {imgRealWidth,imgRealHeight,0.0f} };
	for (int i = 0; i < nImages; i++) {
		posX = x[i]; posY = y[i]; posZ = z[i];

		for (int j = 0; j < 4; j++) {
			posX = R[i][0]*vects[j][0] + R[i][1]*vects[j][1] + R[i][2]*vects[j][2] + posX;
			posY = R[i][3]*vects[j][0] + R[i][4]*vects[j][1] + R[i][5]*vects[j][2] + posY;
			posZ = R[i][6]*vects[j][0] + R[i][7]*vects[j][1] + R[i][8]*vects[j][2] + posZ;

			xMin = (posX < xMin ? posX : xMin);
			yMin = (posY < yMin ? posY : yMin);
			zMin = (posZ < zMin ? posZ : zMin);
			xMax = (posX > xMax ? posX : xMax);
			yMax = (posY > yMax ? posY : yMax);
			zMax = (posZ > zMax ? posZ : zMax);
		}
	}

	const float boxWidth = xMax - xMin;
	const float boxHeight = yMax - yMin;
	const float boxLentgh = zMax - zMin;
	const unsigned int minVoxelGridWidth = ceil(boxWidth/deltaGrid);
	const unsigned int minVoxelGridHeight = ceil(boxHeight/deltaGrid);
	const unsigned int minVoxelGridLength = ceil(boxLentgh/deltaGrid);
	//const unsigned int voxelGridWidth = pow(2, ceil(log2(boxWidth/deltaGrid)));
	//const unsigned int voxelGridHeight = pow(2, ceil(log2(boxHeight/deltaGrid)));
	//const unsigned int voxelGridLength = pow(2, ceil(log2(boxLentgh/deltaGrid)));
	const unsigned int voxelGridWidth = minVoxelGridWidth;
	const unsigned int voxelGridHeight = minVoxelGridHeight;
	const unsigned int voxelGridLength = minVoxelGridLength;
	const unsigned long gridSize = voxelGridWidth * voxelGridHeight * voxelGridLength;

	log_console.infoStream() << "\tpMin = (" << xMin << "," << yMin << "," << zMin << ")";
	log_console.infoStream() << "\tpMax = (" << xMax << "," << yMax << "," << zMax << ")";
	log_console.infoStream() << "\tBox Size : " << boxWidth << "x" << boxHeight << "x" << boxLentgh << " (mm)";

	log_console.infoStream() << "Voxel grid precision set to " << deltaGrid*1000 << " µm";
	log_console.infoStream() << "Minimum grid size : " 
		<< minVoxelGridWidth << "x" << minVoxelGridHeight << "x" << minVoxelGridLength;
	log_console.infoStream() << "Effective grid size : " 
		<< voxelGridWidth << "x" << voxelGridHeight << "x" << voxelGridLength;
	log_console.infoStream() << "Effective grid memory size (unsigned char) : " 
		<< (gridSize > 1024*1024 ? gridSize/(1024*1024) * sizeof(unsigned char) : gridSize/1024 * sizeof(unsigned char))
		<< (gridSize > 1024*1024 ? "MB" : "KB");
	log_console.infoStream() << "Effective images memory size (float) : " 
		<< (imageSize > 1024*1024 ? imageSize/(1024*1024) * sizeof(float) : imageSize/1024*sizeof(float))
		<< (imageSize > 1024*1024 ? "MB" : "KB");
	log_console.infoStream() << "Effective images memory size (unsigned char) : " 
		<< (imageSize > 1024*1024 ? imageSize/(1024*1024) * sizeof(unsigned char) : imageSize/1024*sizeof(unsigned char))
		<< (imageSize > 1024*1024 ? "MB" : "KB");
	log_console.infoStream() << "The programm will need to use at least " 
		<< max((gridSize + imageSize)*sizeof(unsigned char), (sizeof(float)+sizeof(unsigned char))*imageSize)/(1024*1024)
		<< " MB of VRAM.";

	if(max((gridSize + imageSize)*sizeof(unsigned char), (sizeof(float)+sizeof(unsigned char))*imageSize)/(1024*1024) >= 1024) {
		log_console.warnStream() << "The programm will use more then 1GB of VRAM, please check if your GPU has enough memory !";
	}

	CudaUtils::logCudaDevices(log_console);

	int maxDevice;
	cudaGetDeviceCount(&maxDevice);
	CHECK_CUDA_ERRORS(cudaSetDevice(maxDevice-1));	
	CHECK_CUDA_ERRORS(cudaFree(0));	

	//image data
	float *device_float_data, *host_float_data;
	unsigned char *device_char_data, *host_char_data;

	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &host_float_data, nImages*imgWidth*imgHeight*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &host_char_data, nImages*imgWidth*imgHeight*sizeof(unsigned char)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &device_float_data, nImages*imgWidth*imgHeight*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &device_char_data, nImages*imgWidth*imgHeight*sizeof(unsigned char)));

	//copy not pinned memory to pinned memory
	for (int i = 0; i < nImages; i++) {
		memcpy(host_float_data + i*imgWidth*imgHeight, data[i], imgWidth*imgHeight*sizeof(float));
		delete [] data[i];
	}
	delete [] data;

	//copy host pinned memory to device memory
	log_console.info("Copying image data to GPU...");
	CHECK_CUDA_ERRORS(cudaMemcpy(device_float_data, host_float_data, nImages*imgWidth*imgHeight*sizeof(float), cudaMemcpyHostToDevice));

	//call kernel
	log_console.info("[KERNEL] Casting image data to unsigned char.");
	testKernel(nImages, imgWidth, imgHeight, device_float_data, device_char_data);

	//free float array
	log_console.info("Free float image data.");
	CHECK_CUDA_ERRORS(cudaFree(device_float_data));
	
	//CHECK_CUDA_ERRORS(cudaMemcpy(host_char_data, device_char_data, nImages*imgWidth*imgHeight, cudaMemcpyDeviceToHost));
	
	//show results
	//Mat m0(imgHeight, imgWidth, CV_32F, data[40]);
	//m0.convertTo(m0, CV_8UC1);
	//Mat m1(imgHeight, imgWidth, CV_8UC1, host_char_data+40*imgWidth*imgHeight);
	//Image::displayImage(m0);
	//Image::displayImage(m1);

	//namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
	//for (int i = 0; i < nImages; i++) {
		//Mat m1(imgHeight, imgWidth, CV_8UC1, host_char_data+i*imgWidth*imgHeight);
		//Mat m0(imgHeight, imgWidth, CV_32F, host_float_data+i*imgWidth*imgHeight);
		//m0.convertTo(m0, CV_8UC1);
		//imshow("Display window", m1);
		//cvWaitKey(100);
		//cout << i << "/" << nImages << endl;
	//}
	
	//Mat m1(imgHeight, imgWidth, CV_8UC1, host_char_data);
	//VideoWriter writer("img/data_0.avi", CV_FOURCC('M','J','P','G'), 12, m1.size(), false);
	 //for (int i = 0; i < nImages; i++) {
		//Mat m0(imgHeight, imgWidth, CV_8UC1, host_char_data+i*imgWidth*imgHeight);
		//writer << m0;
	 //}
	 //return 0;

	log_console.info("Copying offset and rotation data to GPU...");
	//copy offset data
	float *offsetX_d, *offsetY_d, *offsetZ_d;
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &offsetX_d, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &offsetY_d, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &offsetZ_d, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMemcpy(offsetX_d, x, nImages*sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaMemcpy(offsetY_d, y, nImages*sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaMemcpy(offsetZ_d, z, nImages*sizeof(float), cudaMemcpyHostToDevice));
	//copy rotation data
	float *r1_h, *r2_h, *r3_h, *r4_h, *r5_h, *r6_h, *r7_h, *r8_h, *r9_h;
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &r1_h, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &r2_h, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &r3_h, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &r4_h, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &r5_h, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &r6_h, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &r7_h, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &r8_h, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &r9_h, nImages*sizeof(float)));

	for (int i = 0; i < nImages; i++) {
		r1_h[i] = R[i][0];
		r2_h[i] = R[i][1];
		r3_h[i] = R[i][2];
		r4_h[i] = R[i][3];
		r5_h[i] = R[i][4];
		r6_h[i] = R[i][5];
		r7_h[i] = R[i][6];
		r8_h[i] = R[i][7];
		r9_h[i] = R[i][8];
	}

	float *r1_d, *r2_d, *r3_d, *r4_d, *r5_d, *r6_d, *r7_d, *r8_d, *r9_d;
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &r1_d, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &r2_d, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &r3_d, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &r4_d, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &r5_d, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &r6_d, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &r7_d, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &r8_d, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &r9_d, nImages*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMemcpy(r1_d, r1_h, nImages*sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaMemcpy(r2_d, r2_h, nImages*sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaMemcpy(r3_d, r3_h, nImages*sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaMemcpy(r4_d, r4_h, nImages*sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaMemcpy(r5_d, r5_h, nImages*sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaMemcpy(r6_d, r6_h, nImages*sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaMemcpy(r7_d, r7_h, nImages*sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaMemcpy(r8_d, r8_h, nImages*sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaMemcpy(r9_d, r9_h, nImages*sizeof(float), cudaMemcpyHostToDevice));
	
	//allocate voxels
	log_console.info("Allocating voxel grid to GPU...");
	unsigned char *device_voxel_data;
	unsigned char *host_voxel_data;
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &host_voxel_data, gridSize*sizeof(unsigned char)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &device_voxel_data, gridSize*sizeof(unsigned char)));
	
	//allocate hit counter
	log_console.info("Allocating hit counter to GPU...");
	unsigned char *device_hit_counter;
	unsigned char *host_hit_counter;
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &host_hit_counter, gridSize*sizeof(unsigned char)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &device_hit_counter, gridSize*sizeof(unsigned char)));
	
	//set voxels anb hit counter to 0
	log_console.info("Setting memory to 0...");
	CHECK_CUDA_ERRORS(cudaMemset(device_voxel_data, 0, gridSize*sizeof(unsigned char)));
	CHECK_CUDA_ERRORS(cudaMemset(device_hit_counter, 0, gridSize*sizeof(unsigned char)));

	//compute VNN
	log_console.info("[KERNEL] Computing HOLE FILLING using VNN method...");
	VNNKernel(nImages, imgWidth, imgHeight, 
			deltaGrid, deltaX, deltaY,
			xMin, yMin, zMin,
			voxelGridWidth,  voxelGridHeight,  voxelGridLength,
			offsetX_d, offsetY_d, offsetZ_d,
			r1_d, r2_d, r3_d, r4_d, r5_d, r6_d, r7_d, r8_d, r9_d,
			device_char_data, device_voxel_data, device_hit_counter);

	//copy back voxels
	log_console.info("Done. Copying voxels data back to RAM...");
	CHECK_CUDA_ERRORS(cudaMemcpy(host_voxel_data, device_voxel_data, gridSize*sizeof(unsigned char), cudaMemcpyDeviceToHost));
	
	//copy back hit counter
	log_console.info("Done. Copying hit counter data back to RAM...");
	CHECK_CUDA_ERRORS(cudaMemcpy(host_hit_counter, device_hit_counter, gridSize*sizeof(unsigned char), cudaMemcpyDeviceToHost));
	
	long nHit = 0, sumHitRate = 0;
	unsigned char maxHitRate = 0, currentHitRate;
	for (unsigned int i = 0; i < gridSize; i++) {
		currentHitRate = host_hit_counter[i];

		if(currentHitRate != 0) {
			nHit++;
			sumHitRate += currentHitRate;
			maxHitRate = (currentHitRate > maxHitRate ? currentHitRate : maxHitRate);
		}
	}

	log_console.infoStream() << "Theorical filling rate : " << (float)nImages*imgWidth*imgHeight/gridSize;
	log_console.infoStream() << "Actual filling rate : " << (float)nHit/gridSize;
	log_console.infoStream() << "Mean hit rate : " << (float)sumHitRate/nHit;
	log_console.infoStream() << "MaxHitRate hit rate : " << (unsigned int) maxHitRate;

	log_console.info("Free voxel data on CPU.");
	cudaFreeHost(host_voxel_data);

	log_console.info("Free image data on GPU and CPU.");
	CHECK_CUDA_ERRORS(cudaFree(device_char_data));
	CHECK_CUDA_ERRORS(cudaFreeHost(host_char_data));

	log_console.info("Free hit data on CPU and GPU.");
	cudaFree(device_hit_counter);
	cudaFreeHost(host_hit_counter);
	


	log_console.info("Launching voxel engine...");
	QApplication application(argc,argv);


	VoxelRenderer *VR = new VoxelRenderer(
			voxelGridWidth, voxelGridHeight, voxelGridLength, 
			device_voxel_data,
			0.01, 0.01, 0.01, false, viewerThreshold);

	log_console.info("Computing geometry...");
	VR->computeGeometry();
	
	log_console.info("Done!");

	Viewer viewer;
	viewer.addRenderable(VR);
	viewer.setWindowTitle("viewer");
	viewer.show();
	application.exec();


	//free
	log_console.info("Free remaining data.");
	CHECK_CUDA_ERRORS(cudaFree(device_voxel_data));
	//CHECK_CUDA_ERRORS(cudaFreeHost(host_voxel_data));


	return EXIT_SUCCESS;

	////////////////////////////////////////////////////
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

