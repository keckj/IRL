#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <list>

using namespace std;
using namespace cv;

class Image {

	private:
		list<Mat*> images;
		
	public:
		int loadImageFolder(const char *location);
		int loadImage(const char *name);

		void loadLocalizedUSImages(string const & folderName);

		void computeImageFiltering();
		void computeGradientVectorFlow();
		
		static void displayImage(Mat &m);
		static bool compareDataOrder(string const &str1, string const &str2);
};
