#!/bin/bash

# This script is based on a script from the PHOTOX repository (https://github.com/PHOTOX/photoxrepo, Copyright (c) 2014 PHOTOX)
# to maintain compatibility with other tools there

# Driver script for spectra simulation using the reflection principle.
# One can also add gaussian and/or lorentzian broadening.

# It works both for UV/VIS spectra and photoionization spectra.

# REQUIRED FILES:
# calc_spectrum.py
# extractG09.sh or similar

########## SETUP #####
name=novec4710_wB97_6-31pgs
states=50       # number of excited states
               # (ground state does not count)
istart=1       # Starting index
imax=1000      # number of calculations
#indices="absspec.NO3_xtb-stda.n1000.2020-08-17_13-41-00.76.geoms"	# file with indices of geometries to use. Leave empty for using all geometries from istart to imax
#indices=50.geoms	# file with indices of geometries to use. Leave empty for using all geometries from istart to imax
grep_function="grep_G09_TDDFT" # this function parses the outputs of the calculations
               # It is imported e.g. from extractG09.sh
filesuffix="log" # i.e. "com.out" or "log"
ncores=1      # number of cores used for parallel execution of computationally intensive parts such as error bars calculation

## SETUP FOR SPECTRA GENERATION ## 
gauss=0 # Uncomment for Gaussian broadening parameter in eV, set to 0 for automatic setting
#lorentz=0.1 # Uncomment for Lorentzian broadening parameter in eV
de=0.005     # Energy bin for histograms
decompose=false # Turns on/off decomposition of the spectrum into participating excited states (adds one column per each state to the ouput file)
normalize=false # Normalizes spectra maxima to one when true.
ioniz=false # Set to "true" for ionization spectra (i.e. no transition dipole moments)
##############

# Import grepping functions
# At least one of these files must be present
files=(extractDALTON.sh extractG09.sh extractMNDO.sh extractMOLPRO.sh extractMOPAC.sh extractORCA.sh extractQC.sh extractSTDA.sh extractTERA.sh)
for file in "${files[@]}";do
   if [[ -f "$file" ]];then
      source "$file"
   fi
done

i=$istart
samples=0
rawdata="$name.rawdata.$$.dat"

function getData {
   index=$1
   file=$name.$index.$filesuffix
   if  [[ -f $file ]];then
      $grep_function $file $rawdata $states

      if [[ $? -eq "0" ]];then
         if [[ ! -z $subset ]] && [[ $subset > 0 ]];then
                echo $file >> $rawdata
         fi
         let samples++
         echo -n "$i "
      fi
   fi
}
if [[ -n $indices ]] && [[ -f $indices ]]; then
   mapfile -t subsamples < $indices
   for i in "${subsamples[@]}"
   do
      getData $i
   done
else
   while [[ $i -le $imax ]]
   do
      getData $i
      let i++
   done
fi

echo
echo Number of samples: $samples
if [[ $samples == 0 ]];then
	exit 1
fi

options=" --de $de "
if [[ ! -z $gauss ]];then
   options=" -s $gauss "$options
fi
if [[ ! -z $lorentz ]];then
   options=" -t $lorentz "$options
fi
if [[ ! -z $ncores ]] && (( $ncores > 0 ));then
   options=" -j $ncores "$options
fi
if [[ $decompose = "true" ]];then
   options=" -D "$options
fi
if [[ $normalize = "true" ]];then
   options=" --normalize "$options
fi
if [[ $ioniz = "true" ]];then
   options=" --notrans "$options
fi

python calc_spectrum_v2.py -n $samples -N $states --sigmaalg silverman $options $rawdata
#python calc_spectrum_v2.py -n $samples -N $states --sigmaalg cv --onesigma $options $rawdata

