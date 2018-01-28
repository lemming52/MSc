#!/bin/bash

for i in `ls *.pdf`
do
cp /cygdrive/d/CERN/results/hlthists/$1/$i .
done