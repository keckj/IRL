
#ifndef MHDFILE_H
#define MHDFILE_H

#include <iostream>

class MhdFile {

	public:

		MhdFile(const char *dataFolder, const char *mhdFile);
		~MhdFile(); //loaded data is not freed

		void loadData();
		enum ElementType {NO_TYPE = 0, MET_SHORT, MET_FLOAT, MET_UNKNOWN};
	
		unsigned int getDim() const;
		unsigned int *getSize() const;

		unsigned int getWidth() const;
		unsigned int getHeight() const;
		unsigned int getLength() const;
		
		float* getTransformationMatrix() const;
		float* getOffset() const;

		unsigned long getSamplingFrequency() const; //in Hz
		unsigned long getImagingFrequency() const; //in Hz
		
		void* getData() const;
		ElementType getDataType() const;
		float* getElementSpacing() const;

	
	private:
		unsigned int NDims;
		unsigned int *dimSize;
		float* elementSpacing;

		float *transformationMatrix;
		float *offset;

		unsigned long samplingFrequency;
		unsigned long imagingFrequency;

		std::string elementDataFile;
		ElementType elementType;
		void *data;
};

std::ostream& operator<<(std::ostream& os, const MhdFile& obj);

#endif /* end of include guard: MHDFILE_H */
