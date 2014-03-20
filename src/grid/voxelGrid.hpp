#ifndef VOXELGRID_H
#define VOXELGRID_H

class VoxelGrid {
public:
	
	explicit VoxelGrid(unsigned char *gridData_h, 
			unsigned char *gridData_d,
			const float gridRealWidth, 
			const float gridRealHeight, 
			const float gridRealLength, 
			const float deltaGrid);
	
	explicit VoxelGrid(unsigned char *gridData_h,
			unsigned char *gridData_d,
			const unsigned int gridWidth, 
			const unsigned int gridHeight, 
			const unsigned int gridLength, 
			const float deltaGrid);
	
	virtual ~VoxelGrid ();

	unsigned int width() const;
	unsigned int height() const;
	unsigned int length() const;
	float voxelSize() const;
	unsigned int dataSize() const;
	unsigned char *dataHost() const;
	unsigned char *dataDevice() const;

	unsigned char operator()(unsigned int i, unsigned int j, unsigned int k);

protected:
	unsigned char *gridData_h, *gridData_d;
	const unsigned int gridWidth, gridHeight, gridLength;
	const float deltaGrid;
};


#endif /* end of include guard: VOXELGRID_H */
