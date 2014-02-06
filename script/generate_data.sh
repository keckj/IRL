#!/bin/bash

random_float() {
	local v=$[100 + (RANDOM % 100)]$[1000 + (RANDOM % 1000)]
	v=$[RANDOM % ($1)].${v:1:2}${v:4:3}
	
	echo $v
}

random_int() {

	local v=$[$1 + RANDOM % ($2 - $1)]
	echo $v
}


if [ -f ../data/data.txt ]
then
	rm ../data/data.txt
fi

touch ../data/data.txt

for i in {1..1000}
do
	str=""
	for j in {1..3} 
	do
		r=`random_int 100 200`
		str="$str $r"
	done

	for j in {1..3} 
	do
		r=`random_float 4`
		str="$str $r"
	done
	
	echo $str >> ../data/data.txt
done

