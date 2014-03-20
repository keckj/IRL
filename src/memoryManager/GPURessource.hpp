#ifndef GPURESSOURCE_H
#define GPURESSOURCE_H

template typename<T>
class GpuResource {
public:
	GpuResource(T *data, unsigned int size, bool owner = true, unsigned int deviceId = 0);

	T* data() const;
	unsigned int size() const;
	unsigned int bytes() const;
	bool owner() const;
	unsigned int deviceId() const;
	
	

private:
	T* _data;
	unsigned int _size;
	bool _owner;
	unsigned int _deviceId;
};

template typename<T>
ostream &operator<<(ostream &out, GpuResource<T> resource);


#endif /* end of include guard: GPURESSOURCE_H */
