#!/bin/bash

input=$1

awk '
'NR%2==1' {energy=$1; print energy}
'NR%2==0' {energy=energy/27.211396; tdpx=sqrt(3*$1/(2*energy)); print tdpx,0.0,0.0}
' $input
