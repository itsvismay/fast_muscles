#!/bin/bash
m=250
for c in 5;
do
	for s in 5 10 15 20;
	do
		f="../data/realistic_biceps/${m}modes${c}clusters${s}handles";
		mkdir $f
	done
done

for c in 5;
do
	for s in 5 10 15 20;
	do
		t="../data/realistic_biceps/${m}modes${c}clusters${s}handles/${m}modes${c}clusters${s}handles.txt";
		./../release/elastic $m $c $s &
	done
done
