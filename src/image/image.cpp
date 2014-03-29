

#include "image/image.hpp"
#include "utils/log.hpp"
#include "image/LocalizedUSImage.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <opencv/cv.h>
#include <QtGui>
#include "memoryManager/CPUMemory.hpp"

using namespace cv;
using namespace std;

vector<string> *listdir(const char *dirname) {
	DIR *dp;
	dirent *d;
	vector<string> *vec = new vector<string>;

	dp = opendir(dirname);
	while((d = readdir(dp)) != NULL)
		vec->push_back(d->d_name);

	sort(vec->begin(), vec->end());
	return vec;
}

int Image::loadImage(const char *name) {

	Mat *image = new Mat();

	*image = imread(name, CV_LOAD_IMAGE_GRAYSCALE);

	if(! image->data) {
		log_console.warn("Failed to load image %s .", name);
		return -1;
	}

	images.push_back(image);

	return 0;
}

int Image::loadImageFolder(const char* folder) {

	vector<string> *files = listdir(folder);

	char str[200];
	int nbImg = 0;

	for (vector<string>::iterator it = files->begin(); it != files->end(); it++) {
		if (*it != "." && *it != "..") {
			sprintf(str, "%s%s", folder, it->c_str());
			nbImg += (loadImage(str) + 1);
		}
	}

	log_console.info("Loaded %i images from folder %s", nbImg, folder);

	return nbImg;
}

void Image::computeGradientVectorFlow() {

	log_console.info("Computing Gradient Vector Flow");

	Mat *img = images.front();

	log_console.debug("Loaded image");
	log_console.debug("Size : %i x %i", img->size[0], img->size[1]);
	log_console.debug("Step : %i x %i", img->step[0], img->step[1]);


	//CANNY
	Mat gaussianBlurredImg, canny;
	log_console.debug("Applying Canny filter");
	GaussianBlur(*img, gaussianBlurredImg, Size(9,9), 5.0);
	int lowThreshold = 30;
	int kernel_size = 3;
	Canny(gaussianBlurredImg, canny, lowThreshold, lowThreshold*3, kernel_size);

	canny.copyTo(*img, canny);
	displayImage(*img);

	//THRESHOLD
	Mat thres;
	log_console.debug("Thresholding");
	threshold(canny, thres, 127, 255, THRESH_BINARY_INV);
	displayImage(thres);

	//SOBEL
	log_console.debug("Applying Sobel filter");
	Mat f(thres.size[0], thres.size[1], CV_16S, Scalar::all(0));
	Mat fx, fy, b, c1, c2;

	//add(f, 0, thres, f);
	displayImage(f);

	Sobel(f, fx, CV_16S, 1, 0, 3);
	Sobel(f, fy, CV_16S, 0, 1, 3);

	//convertScaleAbs(sobelY, sobelY);
	//convertScaleAbs(sobelX, sobelX);
	//addWeighted(sobelX, 0.5, sobelY, 0.5, 0, sobel);

}

void Image::displayImage(Mat &m) {

	namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
	imshow( "Display window", m); 
	waitKey(0);

}

void Image::loadLocalizedUSImages(
		string const & folderName, 
		int *nImage, int *imgWidth, int *imgHeight, float *deltaX, float *deltaY,  
		float ***offsets, float ***rotations, float **data,
		bool pageLockedMemory
		) 
{

	DIR *dir;
	struct dirent *ent;
	unsigned int counter = 0;

	dir = opendir(folderName.c_str());

	log_console.infoStream() << "Loading data from folder " << folderName << " .";

	if (dir == NULL) {
		log_console.errorStream() << "Impossible to open folder " <<  folderName << " !";
		throw std::logic_error("Impossible to open data folder.");
	}


	/* On initialise le loader */
	LocalizedUSImage::initialize();

	/* On cherche les *.mhd */
	list<string> mhdFiles;
	while((ent = readdir(dir)) != NULL) {
		if(ent->d_type == DT_REG && (strstr(ent->d_name, ".mhd") != NULL)) {
			mhdFiles.push_back(string(ent->d_name));
			counter++;
		}
	}

	/* On regarde si on en a au moins un */
	if (counter == 0) {
		log_console.errorStream() << "No data was found in folder " <<  folderName << " !";
		throw std::logic_error("No data was found in given folder.");
	}
	else {
		*nImage = counter;
		log_console.infoStream() << "Found " << counter << " mhd files.";
	}
	

#ifdef _SORT
	//On trie les noms de fichiers (cohérence spaciale) 
	log_console.infoStream() << "Sorting filenames...";
	mhdFiles.sort(Image::compareDataOrder);
#endif
	

	/* On charge les données */
	log_console.infoStream() << "Processing data...";

	*offsets = new float*[3];
	for (int i = 0; i < 3; i++) {
		if(pageLockedMemory)
			(*offsets)[i] = CPUMemory::malloc<float>(*nImage, true);
			//cudaMallocHost((void**) (*offsets+i), (*nImage)*sizeof(float));
		else
			(*offsets)[i] = CPUMemory::malloc<float>(*nImage, false);
			//(*offsets)[i] = new float[(*nImage)];
	}
	
	*rotations = new float*[9];
	for (int i = 0; i < 9; i++) {
		if(pageLockedMemory)
			(*rotations)[i] = CPUMemory::malloc<float>(counter, true);
			//cudaMallocHost((void**) (*rotations+i), counter*sizeof(float));
		else
			(*rotations)[i] = CPUMemory::malloc<float>(counter, false);
			//(*rotations)[i] = new float[counter];
	}
	
	unsigned int read_counter = 0;
	bool firstImg = true;

	for (list<string>::iterator it = mhdFiles.begin() ; it != mhdFiles.end(); it++) {
		LocalizedUSImage img(folderName, *it);

		if (img.isOk()) {
			if(firstImg) {
				//get Size information
				*imgWidth = img.getWidth();
				*imgHeight = img.getHeight();
				*deltaX = LocalizedUSImage::getElementSpacing()[0];
				*deltaY = LocalizedUSImage::getElementSpacing()[1];
				
				//allocate data array
				if(pageLockedMemory)
					*data = CPUMemory::malloc<float>((*nImage)*img.getWidth()*img.getHeight(), true);
					//cudaMallocHost((void**) data, (*nImage)*img.getWidth()*img.getHeight()*sizeof(float));
				else
					*data = CPUMemory::malloc<float>((*nImage)*img.getWidth()*img.getHeight(), false);
					//*data = new float[(*nImage)*img.getWidth()*img.getHeight()];

				firstImg = false;
			}

			for (int i = 0; i < 3; i++) {
				(*offsets)[i][read_counter] = img.getOffset()[i];
			}
			
			for (int i = 0; i < 9; i++) {
				(*rotations)[i][read_counter] = img.getRotationMatrix()[i];
			}
			
			memcpy((*data + read_counter*img.getWidth()*img.getHeight()), img.getImageData(), img.getWidth()*img.getHeight()*sizeof(float));

			read_counter++;
		}
		else {
			(*nImage)--;
			log_console.warnStream() << "The file " << *it << "is not valid. It has been skipped.";
		}
	}
	


	//ofstream outfile;
	//outfile.open(folderName + "data.txt", ifstream::out);

	//for(unsigned int i = 0; i < counter; i++) {
	//outfile << x[i] << " \t" << y[i] << " \t" << z[i];
	//for (int j = 0; j < 9; j++) {
	//outfile << "\t" << R[i][j];
	//}

	//outfile << endl;
	//}

	//outfile.close();

	log_console.infoStream() << "Read " << counter << "valid data files.";
	log_console.infoStream() << "Finished to read data from folder " << folderName << ".";
}

bool Image::compareDataOrder(string const & str1, string const & str2) {

	istringstream stream1(str1);
	istringstream stream2(str2);

	int n1, n2;

	while(stream1.get() != '#');
	while(stream2.get() != '#');

	stream1 >> n1;
	stream2 >> n2;

	return n1 < n2;
}
		
		
		
void Image::filter1D(float *data, int nData, int size, float sigma) {

	assert(size % 2 == 1);
	assert(sigma > 0.0f);

	float *kernel = new float[size/2 + 1];

	float coef, sum;
	int i;

	for (i = 0; i <= size/2; i++) {
		coef = 1/sigma * exp(-i*i/(2.0f*sigma*sigma));
		sum += coef;
		kernel[i] = coef;  
	}

	for (i = 1; i <= size/2; i++) {
		sum += kernel[i];
	}
	
	float *old_vals = new float[size/2];
	float current;
	
	for (i = 0; i < size/2; i++) {
		old_vals[i] = data[i];
	}
	
	for (int n = 0; n < nData; n++) {
		if(n < size/2)
			;
		else if(n > nData - 1 -size/2)
			;
		else {
			current = data[n]*kernel[0];

			for (i = 0; i < size/2; i++) {
				current += old_vals[i]*kernel[i+1];
				current += data[n+i+1]*kernel[i+1];
			}
			
			//actualize old values
			for (i = 0; i < size/2; i++) {
				old_vals[i] = (i == size/2 - 1 ? data[n] : old_vals[i+1]);
			}

			data[n] = current/sum;
		}
	}
	
}

QImage *Image::generateParallelSlice(unsigned int nSlice,
	VoxelGrid<unsigned char,PinnedCPUResource,GPUResource> &grid, SliceAxe axe) {

	
	unsigned int sliceWidth=0, sliceHeight=0;

	switch(axe) {
		case(AXE_X):
			sliceWidth = grid.height();
			sliceHeight = grid.length();
			break;
		case(AXE_Y):
			sliceWidth = grid.length();
			sliceHeight = grid.width();
			break;
		case(AXE_Z):
			sliceWidth = grid.width();
			sliceHeight = grid.height();
			//return new QImage(grid.dataHost() + sliceWidth*sliceHeight*sizeof(unsigned char), sliceWidth, sliceHeight, sliceWidth, QImage::Format_Indexed8);
			break;
	}


	unsigned char *data = new unsigned char[sliceWidth*sliceHeight];

	for (unsigned int i = 0; i < sliceHeight; i++) {
		for (unsigned int j = 0; j < sliceWidth; j++) {
					switch(axe) {
						case(AXE_X):
							data[i*sliceWidth+j] = grid(nSlice,j,i);
							break;
						case(AXE_Y):
							data[i*sliceWidth+j] = grid(i,nSlice,j);
							break;
						case(AXE_Z):
							data[i*sliceWidth+j] = grid(j,i,nSlice);
							break;
					}
		}
	}

	return new QImage(data, sliceWidth, sliceHeight, sliceWidth, QImage::Format_Indexed8);
}

void Image::displayQTImage(Mat &m) {
     QWidget window;
     window.resize(800, 600);
     window.show();
     window.setWindowTitle("test");
}
