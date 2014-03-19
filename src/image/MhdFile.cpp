
#include "MhdFile.hpp"
#include "utils/log.hpp"
#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;

MhdFile::MhdFile(const char *dataFolder, const char *mhdFile) :
	NDims(0), dimSize(0), elementSpacing(0),
	transformationMatrix(0), offset(0),
	samplingFrequency(0), imagingFrequency(0),
	elementDataFile(""), elementType(ElementType::NO_TYPE), data(0)

{

	log_console.infoStream() << "Opening mhd file > " << mhdFile;
	ifstream infile;
	
	string filePath(string(dataFolder) + string(mhdFile));
	infile.open(filePath, ifstream::in);

	if(infile.bad() || infile.fail()) {
		log_console.errorStream() << "Error while loading mhd file" << filePath << " !";
		return;
	}
	
	string line, word;
	while (infile.good()){

		getline(infile, line);
		stringstream lineStream(line);	
		lineStream >> word;

		if(word.compare("NDims") == 0) {
			log_console.debug("Found NDims !");
			assert(NDims == 0);
			lineStream >> word; // delete '='
			//get NDims 
			lineStream >> NDims;
		}
		else if(word.compare("TransformMatrix") == 0) {
			log_console.debug("Found transformation matrix !");
			assert(transformationMatrix == 0);
			transformationMatrix = new float[9];
			lineStream >> word; // delete '='
			//get the rotation matrix
			for (int i = 0; i < 9; i++) {
				lineStream >> transformationMatrix[i];
			}
		}
		else if (word.compare("Offset") == 0) {
			log_console.debug("Found offset vector !");
			assert(offset == 0);
			offset = new float[3];
			lineStream >> word; // delete '='
			//get offset
			for (int i = 0; i < 3; i++) {
				lineStream >> offset[i];
			}
		}
		else if (word.compare("ElementSpacing") == 0) {
			log_console.debug("Found element spacing !");
			assert(elementSpacing == 0);
			elementSpacing = new float[3];
			lineStream >> word; // delete '='
			//get element spacing
			for (int i = 0; i < 3; i++) {
				lineStream >> elementSpacing[i];
			}
				
		}
		else if(word.compare("DimSize") == 0) {
			log_console.debug("Found image size !");
			assert(dimSize == 0);
			dimSize = new unsigned int[3];
			lineStream >> word; // delete '='
			//get the size of the image
			for (int i = 0; i < 3; i++) {
				lineStream >> dimSize[i];
			}
		}
		else if(word.compare("ElementType") == 0) {
			log_console.debug("Found element type !");
			assert(elementType == ElementType::NO_TYPE);
			lineStream >> word; // delete '='
			lineStream >> word; // getType 

			//we assume type is known 
			if(word.compare("MET_FLOAT") == 0) {
				log_console.debugStream() << "Type is MET_FLOAT > " << mhdFile << " !";
				elementType = ElementType::MET_FLOAT;
			}
			else if(word.compare("MET_SHORT") == 0) {
				log_console.debugStream() << "Type is MET_SHORT > " << mhdFile << " !";
				elementType = ElementType::MET_SHORT;
			}
			else {
				log_console.critStream() << "Type is UNKNOWN > " << mhdFile << " !";
				elementType = ElementType::MET_UNKNOWN;
			}
		}
		else if(word.compare("SamplingFreqHz") == 0) {
			log_console.debug("Found sampling frequency !");
			assert(samplingFrequency == 0);
			lineStream >> word; // delete '='
			lineStream >> samplingFrequency;
		}
		else if(word.compare("ImagingFreqHz") == 0) {
			log_console.debug("Found imaging frequency !");
			assert(imagingFrequency == 0);
			lineStream >> word; // delete '='
			lineStream >> imagingFrequency;
		}
		else if(word.compare("ElementDataFile") == 0) {
			log_console.debug("Found data file !");
			lineStream >> word; // delete '='
			lineStream.get(); //delete space

			stringstream buffer;
			while(lineStream.peek() != -1) {
				buffer << (char) lineStream.get();
			}

			elementDataFile = buffer.str();
			elementDataFile = string(dataFolder) + elementDataFile;
		}
	}

	infile.close();
}


MhdFile::~MhdFile() {
	delete [] dimSize;
	delete [] elementSpacing;
	delete [] transformationMatrix;
	delete [] offset;
}

void MhdFile::loadData() {

	assert(NDims != 0);
	assert(elementSpacing != 0);
	
	ifstream infile;
	infile.open(elementDataFile, ios::in|ios::binary); //open raw data file

	if(infile.bad() || infile.fail()) {
		log_console.errorStream() << "Error while loading raw data file" << elementDataFile << " !";
		return;
	}

	unsigned long dataCharSize = 1;
	for (unsigned int i = 0; i < NDims; i++) {
		dataCharSize *= dimSize[i];
	}

	switch(elementType) {
		case ElementType::MET_SHORT:
			dataCharSize *= sizeof(short);
			break;
		case ElementType::MET_FLOAT:
			dataCharSize *= sizeof(float);
			break;
		default:
			assert(false);
	}

	char *dataBuffer = (char*) calloc(dataCharSize, 1);

	infile.seekg(0, ios::beg);
	infile.read(dataBuffer, dataCharSize);

	data = (void*) dataBuffer;

	infile.close();
	
	log_console.infoStream() << "Loaded raw data from file " << elementDataFile << " !";
}

ostream& operator<<(ostream& os, const MhdFile& obj) {
	
	os << "::Meta Image::" << endl;
	os << "\tElement data type : ";
	
	switch(obj.getDataType()) {
		case MhdFile::ElementType::NO_TYPE:
			os << "NO_TYPE";
			break;
		case MhdFile::ElementType::MET_FLOAT:
			os << "MET_FLOAT";
			break;
		case MhdFile::ElementType::MET_SHORT:
			os << "MET_SHORT";
			break;
		default:
			os << "Unknown";
			break;
	}
	os << endl;

	os << "\tDimension : " << obj.getDim() << endl;
	os << "\tSize : " 
		<< (obj.getDim() >= 1 ? obj.getWidth() : 0) << " x " 
		<< (obj.getDim() >= 2 ? obj.getHeight() : 0) << " x "
		<< (obj.getDim() >= 3 ? obj.getLength() : 0)
		<< endl;
	
	if(obj.getElementSpacing() != 0) {
		os << "\tElement spacing : " << endl;
		os << "\t\t[" 
			<< (obj.getDim() >= 1 ? obj.getElementSpacing()[0] : 0) << ", " 
			<< (obj.getDim() >= 2 ? obj.getElementSpacing()[1] : 0) << ", " 
			<< (obj.getDim() >= 3 ? obj.getElementSpacing()[2] : 0) << "]"
			<< endl;
	}

	if(obj.getOffset() != 0) {
		os << "\tOffset : " << endl;
		os << "\t\t[" << obj.getOffset()[0] << ", " << obj.getOffset()[1] << ", " << obj.getOffset()[2] << "]" << endl;
	}
	if(obj.getTransformationMatrix() != 0) {
		os << "\tTransformation matrix : " << endl;
		os << "\t\t[" 
			<< obj.getTransformationMatrix()[0] << " " 
			<< obj.getTransformationMatrix()[1] << " " 
			<< obj.getTransformationMatrix()[2] 
			<< "]" << endl;
		os << "\t\t[" 
			<< obj.getTransformationMatrix()[3] << " " 
			<< obj.getTransformationMatrix()[4] << " " 
			<< obj.getTransformationMatrix()[5] 
			<< "]" << endl;
		os << "\t\t[" 
			<< obj.getTransformationMatrix()[6] << " " 
			<< obj.getTransformationMatrix()[7] << " " 
			<< obj.getTransformationMatrix()[8] 
			<< "]" << endl;
	}
	if(obj.getSamplingFrequency() != 0)
		os << "\tSampling frequency : " << obj.getSamplingFrequency() << " Hz" << endl;
	if(obj.getImagingFrequency() != 0)
		os << "\tImaging frequency : " << obj.getImagingFrequency() << " Hz" << endl;

	return os;
}


unsigned int MhdFile::getDim() const {
	return this->NDims;
}
unsigned int *MhdFile::getSize() const {
	return this->dimSize;
}

unsigned int MhdFile::getWidth() const {
	assert(NDims>=1);
	return this->dimSize[0];
}
unsigned int MhdFile::getHeight() const {
	assert(NDims>=2);
	return this->dimSize[1];
}
unsigned int MhdFile::getLength() const {
	assert(NDims>=3);
	return this->dimSize[2];
}


float* MhdFile::getTransformationMatrix() const {
	return this->transformationMatrix;
}

float* MhdFile::getOffset() const {
	return this->offset;
}

void* MhdFile::getData() const {
	return this->data;
}
		
MhdFile::ElementType MhdFile::getDataType() const {
	return this->elementType;
}

float* MhdFile::getElementSpacing() const {
	return this->elementSpacing;
}

unsigned long MhdFile::getSamplingFrequency() const {
	return this->samplingFrequency;
}

unsigned long MhdFile::getImagingFrequency() const {
	return this->imagingFrequency;
}
