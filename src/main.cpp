#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "utils/log.hpp"
#include "image/image.hpp"
#include "image/LocalizedUSImage.hpp"
#include "image.hpp"

using namespace std;
using namespace cv;

int main( int argc, const char* argv[] )
{
	initLogs();

	//Image im;
	//im.loadLocalizedUSImages("data/imagesUS/");
	//im.loadLocalizedUSImages("data/processedImages/");
	//return EXIT_SUCCESS;

	LocalizedUSImage::initialize();
	LocalizedUSImage img("data/processedImages/" , "IQ[data #123 (RF Grid).mhd");

	Mat m(img.getHeight(), img.getWidth(), CV_32F, img.getImageData());

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

