
#pragma once

#include <string>

using namespace std;

class LocalizedUSImage {

	public:
		LocalizedUSImage(string const & dataFolder, string const & mhdFile, bool rawMajor = false);
		~LocalizedUSImage();

		int getWidth() const;
		int getHeight() const;
		
		float* getRotationMatrix() const;
		float* getOffset() const;
		float* getImageData() const;

		bool isRawMajor() const;
		bool isOk() const;

		static float* getElementSpacing() ;
		static float getSamplingFrequency() ; //in Hz
		static float getImagingFrequency() ; //in Hz

		static void initialize();

		
	private:
		int nData;
		int width, height;

		bool okStatus;
		bool rawMajor;

		float *rotationMatrix;
		float *offset;
		float *imageData;

		static float* elementSpacing;
		static long samplingFrequency;
		static long imagingFrequency;

		bool parseMhdFileAndLoadRawData(string const & dataFolder, string const & mhdFile);
};

ostream& operator<<(ostream& os, const LocalizedUSImage& obj);


	
