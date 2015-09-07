#!/bin/bash

for f in *.txt
do
    if grep -Fxq 'results' $f
    then
        mv $f $f.bak
        sed '1,/results/d' $f.bak > $f
        rm $f.bak
    fi
done