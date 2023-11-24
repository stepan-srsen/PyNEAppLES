#!/bin/bash

# Driver script for spectra simulation using the nuclear ensemble approach.
# One can use histogramming or gaussian and/or lorentzian broadening.
# It works both for UV/VIS spectra and photoionization spectra.
# This is a wrapper around the python code with basic features only.

# REQUIRED FILES:
# calc_spectrum_v2.py

##### SETUP #########################################################
input=trans-azobenzene.1-1000.n1000.s7.exc.txt	# the input file with excitation energies and possibly transition dipole moments
samples=1000	# number of geometries
states=7	# number of excited states
		# (ground state does not count)
# TODO: istart=1	# Starting index
# TODO: imax=1000	# number of calculations
# TODO: indices=	# file with indices of geometries to use. Leave empty for using all geometries from istart to imax
gauss=0		# Uncomment for Gaussian broadening parameter in eV, set to 0 for automatic setting
#lorentz=0.1	# Uncomment for Lorentzian broadening parameter in eV
de=0.005	# Energy bin for histograms or resolution for broadened spectra.
decompose=false	# Turns on/off decomposition of the spectrum into participating excited states (adds one column per each state to the ouput file)
normalize=false	# Normalizes spectra maxima to one when true.
ioniz=false	# Set to "true" for ionization spectra (i.e. no transition dipole moments)
ncores=1	# number of cores used for parallel execution of computationally intensive parts such as error bars calculation
#####################################################################

nlines=$(wc -l < $input)
nlines2=$((samples * states))
nlines3=$((2 * samples * states))
if [[ $nlines != $nlines2 && $ioniz = "true" ]] || [[ $nlines != $nlines3 && $ioniz != "true" ]]; then
   echo "WARNING: # of lines in the input does not correspond to the ioniz option and # of samples and states."
   echo "# of lines: $nlines, # of samples: $samples, # of states: $states, ioniz=$ioniz"
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

command="python calc_spectrum_v2.py -n $samples -N $states --sigmaalg silverman $options $input"
#command="python calc_spectrum_v2.py -n $samples -N $states --sigmaalg cv --onesigma $options $input"
echo "executing: $command"
eval $command
