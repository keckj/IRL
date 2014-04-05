
#include "image/LocalizedUSImage.hpp"
#include "utils/log.hpp"
#include <iostream>
#include <fstream>

float* LocalizedUSImage::elementSpacing = new float[3];
long LocalizedUSImage::samplingFrequency = 0;
long LocalizedUSImage::imagingFrequency = 0;

LocalizedUSImage::LocalizedUSImage(string const & dataFolder, string const & mhdFile, bool rawMajor) : 
	okStatus(false), rawMajor(rawMajor), imageData(0)
{
	rotationMatrix = new float[9];
	offset = new float[3];

	okStatus = parseMhdFileAndLoadRawData(dataFolder, mhdFile);
}

LocalizedUSImage::~LocalizedUSImage() {
	delete[] rotationMatrix;
	delete[] offset;
	free(imageData);
}

int LocalizedUSImage::getWidth() const {
	return this->width;
}
int LocalizedUSImage::getHeight() const {
	return this->height;
}

float* LocalizedUSImage::getRotationMatrix() const {
	return this->rotationMatrix;
}

float* LocalizedUSImage::getOffset() const {
	return this->offset;
}

float* LocalizedUSImage::getImageData() const {
	return this->imageData;
}

bool LocalizedUSImage::isRawMajor() const {
	return rawMajor;
}

float* LocalizedUSImage::getElementSpacing() {
	return LocalizedUSImage::elementSpacing;
}

float LocalizedUSImage::getSamplingFrequency() {
	return LocalizedUSImage::samplingFrequency;
}

float LocalizedUSImage::getImagingFrequency() {
	return LocalizedUSImage::imagingFrequency;
}

void LocalizedUSImage::initialize() {
	LocalizedUSImage::elementSpacing[0] = 0.0f;
	LocalizedUSImage::elementSpacing[1] = 0.0f;
	LocalizedUSImage::elementSpacing[2] = 0.0f;
}

bool LocalizedUSImage::parseMhdFileAndLoadRawData(string const & dataFolder, string const & mhdFile) {

	log_console.infoStream() << "Opening mhd file > " << mhdFile;
	ifstream infile;

	infile.open(dataFolder + mhdFile, ifstream::in);

	if(infile.bad() || infile.fail()) {
		log_console.errorStream() << "Error while loading mhd file" << mhdFile << " !";
		return false;
	}

	string line, word, dataFile;
	while (infile.good()){

		getline(infile, line);
		stringstream lineStream(line);	
		lineStream >> word;

		if(word.compare("TransformMatrix") == 0) {
			log_console.debug("Found transformation matrix !");
			lineStream >> word; // delete '='
			//get the rotation matrix
			for (int i = 0; i < 9; i++) {
				lineStream >> rotationMatrix[i];
			}
		}
		else if (word.compare("Offset") == 0) {
			log_console.debug("Found offset vector !");
			lineStream >> word; // delete '='
			//get offset
			for (int i = 0; i < 3; i++) {
				lineStream >> offset[i];
			}
		}
		else if (word.compare("ElementSpacing") == 0) {
			log_console.debug("Found element spacing !");
			lineStream >> word; // delete '='

			//first image
			if(LocalizedUSImage::elementSpacing[0] == 0.0f &&
					LocalizedUSImage::elementSpacing[1] == 0.0f &&
					LocalizedUSImage::elementSpacing[2] == 0.0f) {
				//get element spacing
				for (int i = 0; i < 3; i++) {
					lineStream >> LocalizedUSImage::elementSpacing[i];
				}
				
				log_console.infoStream() << "Set global element spacing to " 
					<< elementSpacing[0] << "\t" 
					<< elementSpacing[1] << "\t"
					<< elementSpacing[2] << ".";
			}
			else { //at least 2nd img
				//verify element spacing
				float spacing;
				for (int i = 0; i < 3; i++) {
					lineStream >> spacing;
					if (spacing != LocalizedUSImage::elementSpacing[i]) {
						log_console.warnStream() << "Not same element spacing between images > " << mhdFile << " !";
					}
				}
			}
		}
		else if(word.compare("DimSize") == 0) {
			log_console.debug("Found image size !");
			lineStream >> word; // delete '='
			//get the size of the image
			lineStream >> width;
			lineStream >> height;
		}
		else if(word.compare("ElementType") == 0) {
			log_console.debug("Found image type !");
			lineStream >> word; // delete '='
			lineStream >> word; // getType 

			//we assume type is float
			if(word.compare("MET_FLOAT") != 0) {
				log_console.critStream() << "Type is not MET_FLOAT > " << mhdFile << " !";
			}
		}
		else if(word.compare("SamplingFreqHz") == 0) {
			log_console.debug("Found sampling frequency !");
			lineStream >> word; // delete '='
			if(LocalizedUSImage::samplingFrequency == 0.0f) {
				lineStream >> LocalizedUSImage::samplingFrequency; //set sampling frequency
			}
			else { //at least 2nd image, compare freqs
				long freq;
				lineStream >> freq;
				if(freq != LocalizedUSImage::samplingFrequency) {
					log_console.warnStream() << "Not same sampling frequency between images > " << mhdFile << " !";
				}
			}
		}
		else if(word.compare("ImagingFreqHz") == 0) {
			log_console.debug("Found imaging frequency !");
			lineStream >> word; // delete '='
			if(LocalizedUSImage::imagingFrequency == 0.0f) {
				lineStream >> LocalizedUSImage::imagingFrequency; //set sampling frequency
			}
			else { //at least 2nd image, compare freqs
				long freq;
				lineStream >> freq;
				if(freq != LocalizedUSImage::imagingFrequency) {
					log_console.warnStream() << "Not same imaging frequency between images > " << mhdFile << " !";
				}
			}
		}
		else if(word.compare("ElementDataFile") == 0) {
			log_console.debug("Found data file !");
			lineStream >> word; // delete '='
			lineStream.get(); //delete space

			stringstream buffer;
			while(lineStream.peek() != -1) {
				buffer << (char) lineStream.get();
			}

			dataFile = buffer.str();
			dataFile = dataFolder + dataFile;
		}

	}

	infile.close();

	//log_console.infoStream() << "Finished to read > " << mhdFile;

	//create data array

	log_console.infoStream() << "Opening corresponding raw data file >" << dataFile<<"<";

	infile.clear(); //clear flags (eof)

	infile.open(dataFile, ios::in|ios::binary); //open raw data file

	if(infile.bad() || infile.fail()) {
		log_console.errorStream() << "Error while loading raw data file" << dataFile << " !";
		return false;
	}

	unsigned long dataCharSize = width*height*sizeof(float);
	char *dataBuffer = (char*) calloc(dataCharSize, 1);

	infile.seekg(0, ios::beg);
	infile.read(dataBuffer, dataCharSize);

	imageData = (float*) dataBuffer;

	log_console.infoStream() << "Finished to read data from > " << dataFile;

	infile.close();
		
	return true;
}

ostream& operator<<(ostream& os, const LocalizedUSImage& obj) {
	
	os << "::Localized US Image::" << endl;
	os << "\tSize : " << obj.getWidth() << " x " << obj.getHeight() << endl;
	os << "\tOffset : " << endl;
	os << "\t\t[" << obj.getOffset()[0] << ", " << obj.getOffset()[1] << ", " << obj.getOffset()[2] << "]" << endl;
	os << "\tRotation matrix : " << endl;
	os << "\t\t[" << obj.getRotationMatrix()[0] << " " << obj.getRotationMatrix()[1] << " " << obj.getRotationMatrix()[2] << "]" << endl;
	os << "\t\t[" << obj.getRotationMatrix()[3] << " " << obj.getRotationMatrix()[4] << " " << obj.getRotationMatrix()[5] << "]" << endl;
	os << "\t\t[" << obj.getRotationMatrix()[6] << " " << obj.getRotationMatrix()[7] << " " << obj.getRotationMatrix()[8] << "]" << endl;
	os << "\tSampling frequency : " << obj.getSamplingFrequency() << " Hz" << endl;
	os << "\tImaging frequency : " << obj.getImagingFrequency() << " Hz" << endl;
	os << "\tElement spacing : " << endl;
	os << "\t\t[" << obj.getElementSpacing()[0] << ", " << obj.getElementSpacing()[1] << ", " << obj.getElementSpacing()[2] << "]";
	os << endl;
	
	return os;
}
		
bool LocalizedUSImage::isOk() const {
	return this->okStatus;
}
