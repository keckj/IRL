#ifndef CPURESSOURCE_H
#define CPURESSOURCE_H

template typename<T>
class CPUResource {
public:
	CPUResource(T *data, unsigned int size, bool owner = true);

	T* data() const;
	unsigned int size() const;
	unsigned int bytes() const;
	bool owner() const;
	
private:
	T* _data;
	unsigned int _size;
	bool _owner;
};

template typename<T>
ostream &operator<<(ostream &out, CPUResource<T> resource);


#endif /* end of include guard: CPURESSOURCE_H */
