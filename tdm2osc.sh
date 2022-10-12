#!/bin/bash

awk '
'NR%2==1' {energy=$1; print energy}
'NR%2==0' {energy=energy/27.211396; tdm=$1^2+$2^2+$3^2; fosc=tdm*2*energy/3; print fosc}
' $1
