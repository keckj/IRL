
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "cuda.h"
#include "cuda_runtime.h"

#include "kernels/kernels.hpp"
#include "utils/log.hpp"
#include "utils/cudaUtils.hpp"
#include "image/image.hpp"
#include "image/LocalizedUSImage.hpp"
#include "image/MhdFile.hpp"
#include "qtgui/mainApplication.hpp"

#include "grid/voxelGrid.hpp"
#include "grid/powerOfTwoVoxelGrid.hpp"
#include "grid/voxelGridTree.hpp"

#include "memoryManager/PinnedCPUResource.hpp"
#include "memoryManager/PagedCPUResource.hpp"
#include "memoryManager/GPUResource.hpp"
#include "memoryManager/GPUMemory.hpp"
#include "memoryManager/CPUMemory.hpp"

using namespace std;
using namespace cv;
using namespace kernel;

int main( int argc, char** argv)
{
	initLogs();

	GPUMemory::init();
	CPUMemory::init();
	
	CPUMemory::display(cout);
	GPUMemory::display(cout);

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
	float **offsets_h, **rotations_h, *float_data_h;
	float dx, dy;
	int w, h;
	
	//Note sur le stockage
	//offset[0-2][numImage] //rotation[0-8][numImage], //data[numImage] 
	//       X Y Z                     R0 ... R8
	
	im.loadLocalizedUSImages(dataSource, &nImages, &w, &h, &dx, &dy, &offsets_h, &rotations_h, &float_data_h, true);
	
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


	//compute bounding box
	float posX, posY, posZ;
	float offsetX, offsetY, offsetZ;
	float xMin, xMax, yMin, yMax, zMin, zMax;

	float inf= std::numeric_limits<float>::infinity();
	xMax = -inf; yMax = -inf; zMax = -inf;	
	xMin = +inf; yMin = +inf; zMin = +inf;	

	log_console.infoStream() << "Computing bounding box...";


	//centres des voxels des 4 coins d'une image US
	// -> d'ou les dx/2, dy/2, dz/2
	float vects[4][3] = {
		{dx/2.0f,dy/2.0f,0.0f}, 
		{imgRealWidth - dx/2.0f,dy/2.0f,0.0f}, 
		{dx/2.0f,imgRealHeight -dy/2.0f ,0.0f}, 
		{imgRealWidth - dx/2.0f,imgRealHeight - dy/2.0f,0.0f} 
	};

	for (int i = 0; i < nImages; i++) {
		offsetX = offsets_h[0][i]; offsetY = offsets_h[1][i]; offsetZ = offsets_h[2][i];

		for (int j = 0; j < 4; j++) {
			posX = rotations_h[0][i]*vects[j][0] + rotations_h[1][i]*vects[j][1] + rotations_h[2][i]*vects[j][2] + offsetX;
			posY = rotations_h[3][i]*vects[j][0] + rotations_h[4][i]*vects[j][1] + rotations_h[5][i]*vects[j][2] + offsetY;
			posZ = rotations_h[6][i]*vects[j][0] + rotations_h[7][i]*vects[j][1] + rotations_h[8][i]*vects[j][2] + offsetZ;

			xMin = (posX < xMin ? posX : xMin);
			yMin = (posY < yMin ? posY : yMin);
			zMin = (posZ < zMin ? posZ : zMin);
			xMax = (posX > xMax ? posX : xMax);
			yMax = (posY > yMax ? posY : yMax);
			zMax = (posZ > zMax ? posZ : zMax);
		
			//printf("\nCPU %f\t%f\t%f\n", posX, posY, posZ);
			
		}
	}

	const float boxWidth = xMax - xMin;
	const float boxHeight = yMax - yMin;
	const float boxLentgh = zMax - zMin;
	const unsigned int minVoxelGridWidth = ceil(boxWidth/deltaGrid);
	const unsigned int minVoxelGridHeight = ceil(boxHeight/deltaGrid);
	const unsigned int minVoxelGridLength = ceil(boxLentgh/deltaGrid);
	
	//create grids and compute upper power of two sides grid 
	//no memory is allocated
	VoxelGrid<unsigned char> minGrid(minVoxelGridWidth, minVoxelGridHeight, minVoxelGridLength, deltaGrid);
	PowerOfTwoVoxelGrid<unsigned char> grid(minGrid);
	
	//print devices informations
	CudaUtils::logCudaDevices(log_console);

	//initialize && free all GPU memory
	int maxCudaDevices;
	CHECK_CUDA_ERRORS(cudaGetDeviceCount(&maxCudaDevices));
	for (int i = 0; i < maxCudaDevices; i++) {
		CHECK_CUDA_ERRORS(cudaSetDevice(i));	
		CHECK_CUDA_ERRORS(cudaFree(0));	
	}
	
	//reserved memory on each GPU
	//images, offsets, rotations (stack already taken into account)
	const unsigned long imageBytes = imageSize * sizeof(unsigned char); 	
	const unsigned long rotationBytes = 9*nImages*sizeof(float);
	const unsigned long offsetBytes = 3*nImages*sizeof(float);
	const unsigned long reservedDataBytes = imageBytes + rotationBytes + offsetBytes;
	

	// needed memory on each GPU 
	unsigned long gridBytes = 2*grid.dataBytes();//two grids to allow compute & mem transfers
	unsigned long hitBytes = grid.dataSize()*sizeof(unsigned char); //the hitgrid is shared between the two device voxel grids
	unsigned long sharedBytes = gridBytes + hitBytes;
	float oneGridToNeededMemoryRatio = (float)grid.dataBytes()/sharedBytes;

	//split in subgrids according to min of GPUs memory left, reserved data bytes, and grid size ratio
	VoxelGridTree<unsigned char> voxelGrid = grid.splitGridWithMaxMemory(
			(GPUMemory::getMinAvailableMemoryOnDevices() - reservedDataBytes)*oneGridToNeededMemoryRatio);
	
	//allocate subgrids on host
	for (auto it = voxelGrid.begin(); it != voxelGrid.end(); it++) {
		(*it)->allocateOnHost();
	}

	//compute how many devices are needed
	unsigned int cudaDevicesNeeded = maxCudaDevices;
	unsigned int meanSubgridsToComputePerDevice = 0;
	
	assert(voxelGrid.nChilds() >= 1);
	while(meanSubgridsToComputePerDevice < 1) {
		meanSubgridsToComputePerDevice = voxelGrid.nChilds()/cudaDevicesNeeded;
		cudaDevicesNeeded--;
	}
	cudaDevicesNeeded++;
	
	//compute how many grids should be computed on each device
	unsigned int *subgridsToComputePerDevice = new unsigned int[cudaDevicesNeeded];
	for (int i = 0; i < maxCudaDevices; i++) {
		if(i < (int)cudaDevicesNeeded)
			subgridsToComputePerDevice[i] = meanSubgridsToComputePerDevice;
		else	
			subgridsToComputePerDevice[i] = voxelGrid.nChilds() % meanSubgridsToComputePerDevice;
	}
	
	log_console.infoStream() << "Voxel grid precision set to " << deltaGrid*1000 << " µm";
	log_console.infoStream() << "Minimum grid size : " 
		<< minGrid.width() << "x" << minGrid.height() << "x" << minGrid.length();
	log_console.infoStream() << "Effective grid size : " 
		<< grid.width() << "x" << grid.height() << "x" << grid.length();
	log_console.infoStream() << "Number of subgrids : " 
		<< voxelGrid.nChilds();
	log_console.infoStream() << "Subgrid size : " 
		<< voxelGrid.subwidth() << "x" << voxelGrid.subheight() << "x" << voxelGrid.sublength();
	log_console.infoStream() << "Effective grid memory size (unsigned char) : " 
		<< toStringMemory(grid.dataBytes());
	log_console.infoStream() << "Subgrid memory size (unsigned char) : " 
		<< toStringMemory(voxelGrid.subgridBytes());
	log_console.infoStream() << "Images memory size (unsigned char) : " 
		<< toStringMemory(imageBytes);
	log_console.infoStream() << "Rotations memory size : " 
		<< toStringMemory(rotationBytes);
	log_console.infoStream() << "Offsets memory size : " 
		<< toStringMemory(offsetBytes);
	log_console.infoStream() << "Device memory needed : " 
		<< toStringMemory(reservedDataBytes + 2*voxelGrid.subgridBytes() + voxelGrid.subgridSize() * sizeof(unsigned char)) 
		<< " on " << cudaDevicesNeeded << " devices !"; 
	log_console.infoStream() << "Grid affectation : ";
	for (int i = 0; i < maxCudaDevices; i++) {
		log_console.infoStream() << "Device " << i << " => " << subgridsToComputePerDevice[i];
	}
	log_console.infoStream() << "Number of inactive devices : " << maxCudaDevices - cudaDevicesNeeded;
	
	//allocate two subgrids on each active devices, and a hitgrid
	//and copy images, rotations, offsets
	int currentGrid = 0;
	for (int i = 0; i < cudaDevicesNeeded ; i++) {
		
		unsigned char *subgrid1 = GPUMemory::malloc<unsigned char>();

		for (j = 0; j < subgridsToComputePerDevice[i]; j++) {
		}
	}
	

	
/*
	//image data
	float *float_data_d;
	unsigned char *char_data_d;

	CHECK_CUDA_ERRORS(cudaMalloc((void**) &float_data_d, nImages*imgWidth*imgHeight*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &char_data_d, nImages*imgWidth*imgHeight*sizeof(unsigned char)));

	//copy host memory to device memory
	log_console.info("Copying float image data to GPU...");
	CHECK_CUDA_ERRORS(cudaMemcpy(float_data_d, float_data_h, nImages*imgWidth*imgHeight*sizeof(float), cudaMemcpyHostToDevice));
	
	//free CPU float array
	log_console.info("Free CPU float image data.");
	CHECK_CUDA_ERRORS(cudaFreeHost(float_data_h));

	//call kernel
	log_console.info("[KERNEL] Casting image data to unsigned char.");
	castKernel(nImages, imgWidth, imgHeight, float_data_d, char_data_d);

	//free float array
	log_console.info("Free GPU float image data.");
	CHECK_CUDA_ERRORS(cudaFree(float_data_d));
	
	log_console.info("Copying offset and rotation data to GPU...");
	//copy offset data
	float **offsets_d = new float*[3];
	for (int i = 0; i < 3; i++) {
		CHECK_CUDA_ERRORS(cudaMalloc((void**) &offsets_d[i], nImages*sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMemcpy(offsets_d[i], offsets_h[i] , nImages*sizeof(float), cudaMemcpyHostToDevice));
	}

	//copy rotation data
	float **rotations_d = new float*[9];
	for (int i = 0; i < 9; i++) {
		CHECK_CUDA_ERRORS(cudaMalloc((void**) &rotations_d[i], nImages*sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMemcpy(rotations_d[i], rotations_h[i], nImages*sizeof(float), cudaMemcpyHostToDevice));
	}

	//TODO
	//free rotation and translation
	//log_console.info("Free CPU offset and rotation data.");
	//for (int i = 0; i < 3; i++) {
		//CHECK_CUDA_ERRORS(cudaFreeHost(offsets_h[i]));
	//}
	//for (int i = 0; i < 9; i++) {
		//CHECK_CUDA_ERRORS(cudaFreeHost(rotations_h[i]));
	//}
	
	//allocate voxels
	log_console.info("Allocating voxel grid to GPU...");
	unsigned char *voxel_data_d, *voxel_data_h;
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &voxel_data_d, gridSize*sizeof(unsigned char)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &voxel_data_h, gridSize*sizeof(unsigned char)));
	
	//allocate hit counter
	log_console.info("Allocating hit counter to GPU...");
	unsigned char *hit_counter_h;
	unsigned char *hit_counter_d;
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) &hit_counter_h, gridSize*sizeof(unsigned char)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &hit_counter_d, gridSize*sizeof(unsigned char)));
	
	//set voxels anb hit counter to 0
	log_console.info("Setting memory to 0...");
	CHECK_CUDA_ERRORS(cudaMemset(voxel_data_d, 0, gridSize*sizeof(unsigned char)));
	CHECK_CUDA_ERRORS(cudaMemset(hit_counter_d, 0, gridSize*sizeof(unsigned char)));
	
	//compute VNN
	log_console.info("[KERNEL] Computing BIN FILLING using VNN method...");
	VNNKernel(nImages, imgWidth, imgHeight, 
			deltaGrid, deltaX, deltaY,
			xMin, yMin, zMin,
			voxelGridWidth,  voxelGridHeight,  voxelGridLength,
			offsets_d,
			rotations_d,
			char_data_d, voxel_data_d, hit_counter_d);
	

	cudaDeviceSynchronize();

	//copy back voxelGrid
	CHECK_CUDA_ERRORS(cudaMemcpy(voxel_data_h, voxel_data_d, gridSize*sizeof(unsigned char), cudaMemcpyDeviceToHost));

	//copy back hit counter
	log_console.info("Done. Copying hit counter data back to RAM...");
	CHECK_CUDA_ERRORS(cudaMemcpy(hit_counter_h, hit_counter_d, gridSize*sizeof(unsigned char), cudaMemcpyDeviceToHost));
	
	log_console.info("Free image char data on GPU.");
	CHECK_CUDA_ERRORS(cudaFree(char_data_d));
	
	long nHit = 0, sumHitRate = 0;
	unsigned char maxHitRate = 0, currentHitRate;
	for (unsigned int i = 0; i < gridSize; i++) {
		currentHitRate = hit_counter_h[i];

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

	log_console.info("Free hit data on CPU and GPU.");
	CHECK_CUDA_ERRORS(cudaFree(hit_counter_d));
	CHECK_CUDA_ERRORS(cudaFreeHost(hit_counter_h));


	log_console.info("Launching gui...");
	MainApplication mainApplication(&grid,true,viewerThreshold);	
	return mainApplication.exec();
	*/
}


	//////////////////////////////////////////////////
	//LocalizedUSImage::initialize();
	//LocalizedUSImage img("data/processedImages/" , "IQ[data #123 (RF Grid).mhd");

	//Mat m(img.getHeight(), img.getWidth(), CV_64F, img.getImageData());

	//double min, max;
	//minMaxIdx(m, &min, &max);
	//cout << "\nvals \t" << min << "\t" << max << endl;
	//Mat hist;
	//int hist_size = 128;
	//float range[] = {(float) min, (float) max};
	//const float *hist_range = {range};
	//calcHist(&m, 1, 0, Mat(), hist, 1, &hist_size, &hist_range, true, false);

	//int hist_w = 512; int hist_h = 400;
	//int bin_w = cvRound( (double) hist_w/hist_size );
	//Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
	//normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	//for( int i = 1; i < hist_size; i++ )
	//{
		//line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
				//Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
				//Scalar( 255, 0, 0), 2, 8, 0  );
	//}

	/// Display
	//namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	//imshow("calcHist Demo", histImage );

	//cout << img;
	//Mat m2,m3;
	//m.convertTo(m2, CV_8UC1);
	//m.convertTo(m3, CV_8UC1);
	//GaussianBlur(m2, m2, Size(9,9), 5.0);
	//int lowThreshold = 30;
	//int kernel_size = 3;
	//Canny(m2, m2, lowThreshold, lowThreshold*3, kernel_size);
	//m2.copyTo(m3, m2);
	//Image::displayImage(m3);

	//log_console.info("END OF MAIN PROGRAMM");

	//return EXIT_SUCCESS;
//}

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

	//MhdFile test("data/irm_femur/","MRIm001_fine_registration_complete.mhd");
	//test.loadData();
	//cout << test << endl;

	 //for (unsigned int i = 0; i < test.getLength(); i++) {
		//Mat m(test.getHeight(), test.getWidth(), CV_16U, (signed short*) (test.getData()) +test.getWidth()*test.getHeight()*i);
		//Image::displayImage(m);
		//cvWaitKey(100);
	 //}
	
	//unsigned int height = test.getHeight(), width = test.getWidth(), length = test.getLength();
	//unsigned long size = height*width*length;
	//unsigned char *data = new unsigned char[size];
	//signed short *sdata = (signed short *) test.getData();

	//for (unsigned int i = 0; i < size; i++) {
		//data[i] = (signed char) (sdata[i]/127);
	//}
	 
	//for (unsigned int i = 0; i < test.getLength(); i++) {
		//Mat m(test.getHeight(), test.getWidth(), CV_8U, data + test.getWidth()*test.getHeight()*i);
		//Image::displayImage(m);
		//cvWaitKey(100);
	 //}


	//return 0;
