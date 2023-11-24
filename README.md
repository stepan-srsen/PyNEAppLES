## Python Nuclear Ensemble Approach for Linear Electronic Spectroscopy (PyNEAppLES)

### About
This repository contains several tools for electronic spectra simulations using the nuclear ensemble approach/method (NEA/NEM). Look into references [1] and [2] for the general description of the NEA. The main program has options for the automatic setting of the broadening parameter (the bandwidth), decomposition of the spectrum into electronic states, calculation of error bars (see ref. [2]), etc.

This repository also includes a program for so-called representative sampling (see ref. [3]) allowing to select a small number of geometries representing the initial-state density by employing a fast exploratory ab initio method. Representative sampling can be used to speed up spectra simulations by reducing the number of expensive high-level calculations.

### Tools description

#### calc_spectrum_v2.py

#### CalcSpectrumV2.sh

Bash wrapper around the `calc_spectrum_v2.py` python code with basic features only.

#### osc2tdm.sh

#### tdm2osc.sh

#### repre_sample_1D.py

An older version of representative sampling. It optimizes directly the electronic spectrum at the exploratory level of theory. This code is dependent on `calc_spectrum_v2.py`. Help can be invoked by calling the program with the `-h` switch.

Example call for 1000 samples and 5 states asking for the optimization of a representative sample of 20 geometries using KL diverence and 32 optimization jobs with 2000 cycles on 16 CPU cores. Parameters `--mine` and `--maxe` define the optimized region of the spectrum.
```
python repre_sample_1D.py -n 1000 -N 5 -S 20 --mine 1.7 --maxe 5.4 -c 2000 -j 16 -J 32 --pdfcomp KLdiv input.txt > output.txt
```

The program prints a file with the indices of the selected geometries (non-pythonic indexing, that is, 1st geometry has index 1) which can be further recalculated at a higher level of theory and processed with e.g. `ProcessSpectraV2.sh` or `calc_spectrum_v2.py` scripts. The program also prints the full spectrum and the spectrum of the representative subset. Overall sample statistics of the optimization jobs such as divergences are written to the standard output, that is, to `output.txt` in the example above. The results of the individual optimization jobs are stored in a subfolder.

The `repre_sample_2D.py` replaces this program as it should be in principle able to also perform optimization in 1D but it has not been properly tested for this usecase.

#### repre_sample_2D.py

Newer version of representative sampling code. As opposed to `repre_sample_1D.py`, it is a standalone program and it performs the optimization in the 2D space of excitation energies and transition probabilities calculated at the exploratory level of theory. Help can be invoked by calling the program with the `-h` switch.

Example call for 1000 samples and 5 states asking for the optimization of a representative sample of 20 geometries using KL diverence and 32 optimization jobs with 2000 cycles on 16 CPU cores. The `-w` switch turns on weighting of the distribution by spectroscopic significance (~E*tdm^2) during the optimization.
```
python repre_sample_2D.py -n 1000 -N 5 -S 20 -c 2000 -j 16 -J 32 -w --pdfcomp KLdiv input.txt > output.txt
```

The program prints a file with the indices of the selected geometries (non-pythonic indexing, that is, 1st geometry has index 1) which can be further recalculated at a higher level of theory and processed with e.g. `ProcessSpectraV2.sh` or `calc_spectrum_v2.py` scripts. The program also prints the full 2D distribution and the distribution of the representative subset both as a data file and in the form image file. Overall sample statistics of the optimization jobs such as divergences are written to the standard output, that is, to `output.txt` in the example above. The results of the individual optimization jobs are stored in a subfolder.

Note that PDF divergences based on cumulative distribution functions are not (currently) suitable for optimization in the 2D case. Therefore, Kullback-Leibler divergence (`KLdiv`), Jensen-Shannon divergence (`JSdiv`) or Residual sum of squares (`RSS`) are currently recommended. While KL divergence has a nice interpretation (information loss when approximating one PDF with another) and corresponds to maximum likelihood estimation, it is a bit difficult to work with as both PDFs have to be defined on the same space.

### References
[1] Š. Sršeň, D. Hollas and P. Slavíček, Phys. Chem. Chem. Phys., 2018, 20, 6421–6430, DOI: [10.1039/C8CP00199E](https://doi.org/10.1039/C8CP00199E)  
[2] Š. Sršeň, J. Sita, P. Slavíček, V. Ladányi and D. Heger, J. Chem. Theory Comput., 2020, 16, 6428–6438, DOI: [10.1021/acs.jctc.0c00579](https://doi.org/10.1021/acs.jctc.0c00579)  
[3] Š. Sršeň and P. Slavíček, J. Chem. Theory Comput., 2021, 17, 6395–6404, DOI: [10.1021/acs.jctc.1c00749](https://doi.org/10.1021/acs.jctc.1c00749)
