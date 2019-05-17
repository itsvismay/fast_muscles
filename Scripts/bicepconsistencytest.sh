#!/bin/bash
m=50
for c in 10 20 30;
do
	for s in 10 20;
	do
		f="../data/realistic_upper/${m}modes${c}clusters${s}handles";
		mkdir $f
	done
done

for c in 10 20 30;
do
	for s in 10 20;
	do
		t="../data/realistic_upper/${m}modes${c}clusters${s}handles/${m}modes${c}clusters${s}handles.txt";
		./../release/elastic $m $c $s &
	done
done
