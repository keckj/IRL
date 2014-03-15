#ifndef VOXELGRID_H
#define VOXELGRID_H

class VoxelGrid {
public:
	
	explicit VoxelGrid(unsigned char *gridData,
			const float gridRealWidth, 
			const float gridRealHeight, 
			const float gridRealLength, 
			const float deltaGrid);
	
	explicit VoxelGrid(unsigned char *gridData,
			const unsigned int gridWidth, 
			const unsigned int gridHeight, 
			const unsigned int gridLength, 
			const float deltaGrid);
	
	~VoxelGrid ();

	unsigned int width() const;
	unsigned int height() const;
	unsigned int length() const;
	unsigned int voxelSize() const;
	unsigned int dataSize() const;

private:
	unsigned char *gridData;
	const unsigned int gridWidth, gridHeight, gridLength;
	const float deltaGrid;
};


#endif /* end of include guard: VOXELGRID_H */
