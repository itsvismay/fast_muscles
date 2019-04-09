#!/bin/bash
m=50
for c in 4 5 6;
do
	for s in 4 5 6;
	do
		f="../data/simple_muscle/${m}modes${c}clusters${s}handles";
		mkdir $f
	done
done

for c in 4 5 6;
do
	for s in 4 5 6;
	do
		t="../data/simple_muscle/${m}modes${c}clusters${s}handles/${m}modes${c}clusters${s}handles.txt";
		./../release/elastic $m $c $s &
	done
done
