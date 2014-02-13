

#ifndef _VOXEL_
#define _VOXEL_

#include "renderable.h"

//Voxel contenant un valeur de type T.
//T doit être copiable (attention aux références).

template<typename T>
class Voxel
{
	public:
		Voxel(T value = 0, bool active = false);
		Voxel(Voxel<T> const & other);
		virtual ~Voxel() {};

		virtual Voxel& operator=(const Voxel& source);

	protected:
		T value;
		bool active;

	public:
		T getValue();
		bool isActive();
		void setActive(bool active);
};

#include "voxel.tpp"

#endif

