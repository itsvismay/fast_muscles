#!/bin/bash
m=100
for c in 10;
do
	for s in 15 20;
	do
		f="../data/realistic_biceps/${m}modes${c}clusters${s}handles";
		mkdir $f
	done
done

for c in 10;
do
	for s in 15 20;
	do
		t="../data/realistic_biceps/${m}modes${c}clusters${s}handles/${m}modes${c}clusters${s}handles.txt";
		./../release/elastic $m $c $s &
	done
done
