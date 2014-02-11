

#include "image.hpp"
#include "log.hpp"
#include "LocalizedUSImage.hpp"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <opencv/cv.h>

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

void Image::loadLocalizedUSImages(string const & folderName) {
		
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
			log_console.infoStream() << "Found " << counter << " mhd files.";
			log_console.infoStream() << "Processing raw data...";
		}

		/* On trie les noms de fichiers (cohérence spaciale) */
		mhdFiles.sort(Image::compareDataOrder);

		/* On charge les données */
		float *x = new float[counter];
		float *y = new float[counter];
		float *z = new float[counter];
		float **R = new float*[counter];
		counter = 0;
		
		for (list<string>::iterator it = mhdFiles.begin() ; it != mhdFiles.end(); it++) {
				LocalizedUSImage img(folderName, *it);
				x[counter] = img.getOffset()[0];
				y[counter] = img.getOffset()[1];
				z[counter] = img.getOffset()[2];
				R[counter] = new float[9];

				for (int i = 0; i < 9; i++) {
					R[counter][i] = img.getRotationMatrix()[i];
				}

				counter++;
		}

		
		ofstream outfile;
		outfile.open(folderName + "data.txt", ifstream::out);

		for(unsigned int i = 0; i < counter; i++) {
			outfile << x[i] << " \t" << y[i] << " \t" << z[i];
			for (int j = 0; j < 9; j++) {
				outfile << "\t" << R[i][j];
			}

			outfile << endl;
		}

		outfile.close();
		

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
