# -*- coding: utf-8 -*-
"""
Program for spectra calculation using the reflection principle including error bars and other features.

@author: Stepan Srsen
"""
# This program is based on a script from the PHOTOX repository (https://github.com/PHOTOX/photoxrepo, Copyright (c) 2014 PHOTOX)
# to maintain compatibility with other tools there

from argparse import ArgumentParser
import math
import sys
import numpy as np
import time
import os
#import kde5
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, RepeatedKFold
#from itertools import chain
#import multiprocessing
from joblib import Parallel, delayed, cpu_count
import scipy.stats
#from sklearn.externals.joblib import cpu_count
#import sklearn
import datetime

def read_cmd(parser=None, parse=True):
    """Function for command line parsing."""
    if parser is None:
        parser = ArgumentParser(description='Process spectra.')
    parser.add_argument('infile', help='Input file.')
    parser.add_argument('-n', '--nsamples', type=int, default=1,
                        help='Number of samples.')
    parser.add_argument('-N', '--nstates', type=int, default=1,
                        help='Number of excited states (ground state not included).')
    parser.add_argument('-d', '--de', type=float, default=0.02,
                        help='Bin step in eV. Default = 0.02 ')
    parser.add_argument('-D', '--decompose', action="store_true", default=False,
                        help='Prints the spectrum for each state separately as well.')
    parser.add_argument('-s', '--sigma', type=float, default=-1.0,
                        help='Parameter for Gaussian broadening. Float number for direct setting, negative values for turning off, 0 for automatic setting.')
    parser.add_argument('--onesigma', action="store_true", default=False,
                        help='Optimize one sigma value for all electronic states. Otherwise it selects one sigma per each state.')
    parser.add_argument('-a', '--sigmaalg', choices=['silverman', 'cv', 'dev'], default='silverman',
                        help='Method for setting the Gaussian broadening parameter.')
    parser.add_argument('-t', '--tau', type=float, default=0.0,
                        help='Parameter for Lorentzian broadening.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Activate verbose mode.')
    parser.add_argument('--mine', type=float, default=0.0,
                        help='Minimal energy of the spectrum in eV. Default = 0 for automatic setting.')
    parser.add_argument('--maxe', type=float, default=0.0,
                        help='Maximal energy of the spectrum in eV. Default = 0 for automatic setting. -1 for the minimal energy of the highest state.')
    parser.add_argument('--normalize', action="store_true", default=False,
                        help='Normalize maximum to one.')
    parser.add_argument('--notrans', action="store_true", default=False,
                        help='No transition dipole moments. Returns density of states. Useful for ionizations.')
    parser.add_argument('-e', '--ebars', type=float, default=0.0,
                        help='Calculate error bars / confidence intervals with given confidence from interval (0,1).'
                        + ' Alternatively, it is possible to set it to negative values for multiples of standard deviation, e.g. -2 means 2 standard deviations.')
    # TODO: implement ebar options to class init functions
    parser.add_argument('--eassym', action='store_true',
                        help='Calculate the error bars assymetrically if possible with given algorithm.')
    parser.add_argument('--ealg', choices=['cbb', 'bootstrap', 'subsample', 'jackknife', 'sqrtn'], default='bootstrap',
                        help='Method for error bars calculation.')
    parser.add_argument('-j', '--ncores', type=int, default=1,
                        help='Number of cores for parallel execution of computatinally intensive subtasks:'
                        + ' cross-validation bandwidth setting, error bars, geometry reduction.')
    if parse:
        return parser.parse_args()
    return parser

# TODO: write default values to help
# TODO: generalize representative geometries, nicer output
# TODO: automatic block size / nblocks for error bars
# TODO: repair normalize + check reading and processing data for --notrans
# TODO: reduce dimensionality of kernels variables, exc. state dimm not needed?
# TODO: calc kernel2 only when needed
# TODO: check sqrtn
# TODO: get rid of unnecessary class variables
# TODO: number of cores from cluster environment
# TODO: replace loops in divergence functions
# TODO: write script for selecting geometries from movie using indices from reduction
# TODO: solve reduction + error bars
# TODO: solve error for error bars for big data (>=40k geoms) in parallel processing
# TODO: remove nsamples or nsamples0

def weighted_dev(values, weights, corrected=True):
    """Calculates weighted standard deviation."""
    # equals to np.cov(samples, aweights=weights) for corrected version
    # and np.cov(samples, aweights=weights, bias=True) for uncorrected version
    nweights = weights/np.sum(weights)
    mean = np.sum(np.multiply(values, nweights))
    variance = np.sum(np.multiply((values-mean)**2, nweights))
    if corrected:
        # corrected estimator of weighted std from unbiased variance estimator
        # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
        # http://www.statsmodels.org/dev/generated/statsmodels.stats.weightstats.DescrStatsW.html
        variance /= (1-np.sum(nweights**2))
    return math.sqrt(variance)

def weighted_quantile(samples, weights, percentages, sorted=False):
    """Calculates weighted quantiles."""
# TODO: check repeating values and https://en.wikipedia.org/wiki/Percentile#The_weighted_percentile_method
    if isinstance(percentages, float) or isinstance(percentages, int):
        percentages=[percentages]
    if not sorted:
        indices = np.argsort(samples)
        samples = samples[indices]
        weights = weights[indices]
    cumsum = np.cumsum(weights)
    cumsum /= cumsum[-1]
    quantiles = []
    for percentage in percentages:
        index_lb = len(cumsum[cumsum<=percentage])-1
        if cumsum[index_lb]==percentage:
            quantiles.append(samples[index_lb])
            continue
        quantiles.append(samples[index_lb]*(cumsum[index_lb+1]-percentage)/(cumsum[index_lb+1]-cumsum[index_lb])
                         + samples[index_lb+1]*(percentage-cumsum[index_lb])/(cumsum[index_lb+1]-cumsum[index_lb]))
    if len(quantiles)==1:
        return quantiles[0]
    else:
        return quantiles

def silverman(samples, weights=None, robust=False):
    """Calculates the bandwidth by using Silverman's rule of thumb."""
    if weights is None:
        dev = np.std(samples)
        norm = samples.size
        if robust:
            IQR = np.subtract.reduce(np.quantile(samples, [0.75,0.25]))
    else:
        # import matplotlib.pyplot as plt
        # y,x = np.histogram(weights, bins=200)
        # print(x)
        # print(y)
        # plt.figure()
        # # plt.xlim(left=0)
        # plt.plot(x[1:],y)
        # plt.show()
        dev = weighted_dev(samples, weights, corrected=True)
        # Kish effective sample size
        # https://en.wikipedia.org/wiki/Effective_sample_size#weighted_samples
        # https://docs.displayr.com/wiki/Design_Effects_and_Effective_Sample_Size#Kish.27s_approximate_formula_for_computing_effective_sample_size
        # https://docs.displayr.com/wiki/Design_Effects_and_Effective_Sample_Size
        norm = np.sum(weights)**2/np.sum(weights**2)
        if robust:
            IQR = np.subtract.reduce(weighted_quantile(samples, weights, [0.75,0.25]))
    if robust:
        A = min(dev, IQR/1.349)
        h = 0.9 * A * norm ** (-1./5.)
    else:
        h = (4. / 3.)**(1./5.) * dev * norm**(-1./5.)
    return h

def cv(samples, weights=None, lowerbound=None, upperbound=None, n_jobs=-1):
    """Calculates the bandwidth by using cross validation."""
    # TODO: improve silverman's estimate by averaging over states for --onesigma
    silverman_lcoef = 0.1
    silverman_ucoef = 1.1
    max_it = 20
    npoints = 8
#    bound_coef = 1.3
#    ratio_thr = 1e-1
#    n_splits_coef = 0.15
#    n_repeats_coef = 0.15
#    bound_coef = 1.005
    ratio_thr = 1e-2
    n_splits_coef = 0.05
# necessity of large number of repeats becouse of unequally distributed data (weights) in folds?
    n_repeats_coef = n_splits_coef
#    silverman_ucoef/silverman_lcoef = (1+ratio_thr)**((npoints-1)**i/2^(i-1))
    silv = silverman(samples, weights)
    if lowerbound is None:
        lowerbound = silverman_lcoef * silv 
    if upperbound is None:
        upperbound = silverman_ucoef * silv
    estimator = KernelDensity
    samples = samples[:, None]
    print("Calculating cv...")
    for i in range(max_it):
        start_time = time.time()
        ratio = (upperbound/lowerbound)**(1/(npoints-1))
        n_splits = min(int(round(n_splits_coef/(ratio-1)))+5, samples.size)
        n_repeats = min(int(round(n_repeats_coef/(ratio-1)))+3, samples.size)
        print('STEP', i+1, ':', 'ratio', ratio, 'n_splits', n_splits,
              'n_repeats', n_repeats, 'lowerbound', lowerbound, 'upperbound', upperbound)
        grid = GridSearchCV(estimator(rtol=1e-3), {'bandwidth': np.geomspace(lowerbound, upperbound, num=npoints, endpoint=True)},
                            cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats), n_jobs=n_jobs, verbose=0)
        grid.fit(samples, sample_weight=weights)
        opt = grid.best_params_['bandwidth']
        index = grid.best_index_
        print('STEP', i+1, ':', 'opt', opt, 'index', index, 'time', round(time.time()-start_time))
        if index == npoints-1:
            print('INFO: shifting up...')
            lowerbound *= ratio**(npoints//2)
            upperbound *= ratio**(npoints//2)
            continue
        if index == 0:
            print('INFO: shifting down...')
            lowerbound /= ratio**(npoints//2)
            upperbound /= ratio**(npoints//2)
            if lowerbound <= 0:
                print('ERROR: nonpositive lowerbound during optimization.')
                return False
            continue
        if ratio < (1+ratio_thr):
            break
#        ratio = (upperbound - lowerbound) / (npoints-1)
#        ratio = (upperbound/lowerbound)**(1/(npoints-1))
#        n_splits = int(round(3 + n_splits_coef*opt/ratio))

#        n_repeats = int(round(1 + n_repeats_coef*opt/ratio))
#        lowerbound = opt - bound_coef*ratio
#        upperbound = opt + bound_coef*ratio
#        lowerbound = opt / (bound_coef*ratio)
#        upperbound = opt * (bound_coef*ratio)
        lowerbound = opt / ratio
        upperbound = opt * ratio
        if i == max_it - 1:
            print('WARNING: max number of iterations reached, cv did not converge.')
#    print(grid.cv_results_)
    print("Found optimal bandwidth:", opt)
    return opt

class Spectrum:
    """Basis spectrum class for reflection principle without broadening"""

    def __init__(self, nsamples, nstates, deltaE, normalize, notrans, ncores, verbose, minE, maxE, decompose):
        self.trans = np.empty((nsamples, nstates, 3))
        self.intensity = []
        self.intensities = None
        self.exc = np.empty((nsamples, nstates))
        self.energies = []
        self.samples = []
        self.nsamples = nsamples
        self.nsamples0 = nsamples
        self.nstates = nstates
        self.normalize = normalize
        self.notrans = notrans
        self.verbose = verbose
        self.minE = minE
        self.maxE = maxE
        self.decompose = decompose
        self.de = deltaE # in eV
        self.ncores = ncores
        self.pid = os.getpid()

        self.infile = None
        self.time = None
        # self.npoints = None # only for sqrtn errorbars
        self.acs = None
        self.acs_indices = None
        self.lb = None
        self.up = None

    def read_data(self, infile):
        self.infile = infile
        self.time = datetime.datetime.now()
        with open(self.infile, "r") as f:
            i = 0 #line
            j = 0 #sample
            k = -1 #state
            for line in f:
                if (i % 2 == 1 and not self.notrans):
                    temp = line.split()
                    try:
                  # assigning transition dipole moments as a tuple
                        self.trans[j][k] = (float(temp[0]), float(temp[1]), float(temp[2]))
                    except:
                        print("Error: Corrupted line "+str(i+1)+" in file "+self.infile)
                        print("I expected 3 columns of transition dipole moments, got:")
                        print(line)
                        sys.exit(1)
                else:
                    k += 1
                    if k == self.nstates:
                        k = 0
                        j += 1
                    if j >= self.nsamples0:
                        if line.strip() != "":
                            print("Error: Number of transitions in the input file is bigger than the number of samples multiplied by the number of states.")
                            sys.exit(1)
                        break
                    try:
                        self.exc[j][k] = float(line)
                    except:
                        print("Error when reading file "+self.infile+" on line: "+str(i+1))
                        print("I expected excitation energy, but got:" + line)
                        sys.exit(1)
                i += 1
            if (i != 2*self.nsamples0*self.nstates and not self.notrans) or (i != self.nsamples0*self.nstates and self.notrans):
                print("Error: Number of transitions in the input file is smaller than the number of samples multiplied by the number of states.")
                sys.exit(1)

        if not self.minE:
            self.minE = max(np.min(self.exc)-1.0, self.de)
        if self.maxE == -1:
            self.maxE = np.min(self.exc[:, -1])
        elif not self.maxE:
            self.maxE = np.max(self.exc)+1.0
        if self.verbose:
            print('minE:', self.minE, ', maxE:', self.maxE)

    def calc_energies(self):
        npoints = int(math.floor((self.maxE-self.minE)/self.de))+1
        self.energies = self.minE + np.arange(npoints)*self.de
#        self.energies = self.maxE - np.arange(npoints)*self.de
        
    def trans2acs(self, normalize=False):
        if normalize:
            self.acs = np.ones(self.exc.shape)
        else:
            # Some constants
            EPS = 8.854188e-12   # permitivity of vacuum
            HPRIME = 6.626070e-34/(2*math.pi)   # reduced Planck constant
            C = 299792458   # speed of light
            AUtoCm = 8.478354e-30   # dipole moment a.u. to SI C*m
            COEFF = math.pi * AUtoCm**2 * 1e4 /(3 * HPRIME * EPS * C)
            
            trans2 = np.power(self.trans,2)
            trans2 = np.sum(trans2,axis=2)
            self.acs = COEFF*np.multiply(trans2, self.exc)
            
    def acs2indices(self):       
        indices = np.subtract(self.exc, self.minE)
#        indices = np.subtract(self.maxE, self.exc)
        np.multiply(indices, 1/self.de, out=indices)
        np.rint(indices, out=indices)
        self.acs_indices = indices.astype(int)
        
    def indices2intensity(self, samples=None):
        if samples is None:
            samples = np.arange(self.nsamples0)
        nsamples = len(samples)
        acs = self.acs/(nsamples*self.de)
        intensity = np.zeros((len(self.energies)))
        for i in samples:
            for j in range(self.nstates):
                intensity[self.acs_indices[i,j]] += acs[i,j]
        return intensity
    
    def indices2states(self, samples=None):
        if samples is None:
            samples = np.arange(self.nsamples0)
        nsamples = len(samples)
        acs = self.acs/(nsamples*self.de)
        intensities = np.zeros((len(self.energies), self.nstates))
        for i in samples:
            for j in range(self.nstates):
                intensities[self.acs_indices[i,j], j] += acs[i,j]
        return intensities
    
    def indices2npoints(self, samples=None):
        if samples is None:
            samples = np.arange(self.nsamples0)
        npoints = 1e-6*np.ones((len(self.energies)))
        for i in samples:
            for j in range(self.nstates):
                npoints[self.acs_indices[i,j]] += 1
        return npoints
    
    def calc_spectrum(self, samples=None):
        self.calc_energies()
        self.trans2acs(normalize=self.notrans)
        self.acs2indices()
        self.intensity = self.indices2intensity(samples)
        if self.decompose:
            self.intensities = self.indices2states(samples)
        # self.npoints = self.indices2npoints(samples) # only for sqrtn errorbars
        return self.intensity

    def recalc_spectrum(self, samples=None):
        self.intensity = self.indices2intensity(samples)
        return self.intensity

    def bootstrap_worker(self):
        samples = np.random.choice(np.arange(self.nsamples), size=self.nsamples, replace=True)
        intensity = self.recalc_spectrum(samples=samples)
#       print("worker",index)
        return intensity

    def jackknife_worker(self, index):
        samples = np.concatenate((np.arange(index), np.arange(index+1, self.nsamples)))
        self.nsamples -= 1
        intensity = self.recalc_spectrum(samples=samples)
        self.nsamples += 1
        return intensity

    def subsampling_worker(self, index, nblocks):
        bsize = math.floor(self.nsamples/nblocks)
        remainder = self.nsamples - nblocks*bsize
        if index < remainder:
            bsize += 1
            start = index*bsize
        else:
            start = remainder*(bsize+1) + (index-remainder)*bsize
        samples = np.arange(start, start + bsize)
#        print(samples)
        intensity = self.recalc_spectrum(samples=samples)
        return intensity

    def cbb_worker(self, nblocks):
        bsize = math.floor(self.nsamples/nblocks)
        remainder = self.nsamples - nblocks*bsize
        subsamples = []
        rng = np.arange(self.nsamples)
        for i in range(nblocks):
            rndint = np.random.choice(rng, size=None, replace=True)
            if i < remainder:
                ibsize = bsize + 1
            else:
                ibsize = bsize
            subsamples.append(rng[rndint:rndint+ibsize])
            subsamples.append(rng[0:max(ibsize-self.nsamples+rndint, 0)])
        samples = np.concatenate(subsamples)
        intensity = self.recalc_spectrum(samples=samples)
        return intensity

    def calc_errorbars(self, conf=0.95, assym=False, alg="cbb"):
        # TODO: add central limit theorem (CLT)/asymptotic normality, 
        # i.e. spectrum is an average so estimate average variance, avg over sample vs avg over states and samples
        # remove sqrtn (maybe eq. to CLT?) 
        # jackknife for bias?
        algs = ['cbb', 'bootstrap', 'subsample', 'jackknife', 'sqrtn']
        if alg not in algs:
            print('WARNING: Unknown algorithm for error bars calculation. Skipping.')
            return False
        if assym and not (conf > 0 and conf < 1):
            print('WARNING: Assymetric error bars available only for confidence in (0,1) interval. Skipping.')
            return False

        if alg == 'sqrtn':
            if assym:
                print('WARNING: Assymetric error bars not available for sqrtn algorithm. Skipping.')
                return False
            if self.npoints is None:
                print('ERROR: sqrtn algorithm not implemented')
                return False
            stds = np.divide(self.intensity, np.sqrt(self.npoints))
        else:
            intensity = np.copy(self.intensity)
            with Parallel(n_jobs=self.ncores, verbose=11*int(self.verbose)) as parallel:
                if alg == 'cbb':
                    # should be improved
                    nblocks = round(self.nsamples**(2/3))
                    if options.verbose:
                        print('INFO: nblocks in cbb:', nblocks)
                    intensities = parallel(delayed(self.cbb_worker)(nblocks) for i in range(1000))
                elif alg == 'bootstrap':
                    intensities = parallel(delayed(self.bootstrap_worker)() for i in range(1000))
                elif alg == 'jackknife':
                    intensities = parallel(delayed(self.jackknife_worker)(i) for i in range(self.nsamples))
                elif alg == 'subsample':
                    nblocks = round(self.nsamples**(1/2))
                    if options.verbose:
                        print('INFO: nblocks in subsample:', nblocks)
                    intensities = parallel(delayed(self.subsampling_worker)(i, nblocks) for i in range(nblocks))
#            self.intensitycheck = np.mean(intensities, axis=0)
            self.intensity = intensity
            if assym == False:
                if options.verbose:
                    print('INFO: calculating symmetric error bars with confidence', conf, 'and', alg, 'algorithm.')
                stds = np.std(intensities, axis=0)
                if alg == 'subsample':
                    stds /= math.sqrt(nblocks)
                if conf < 0:
                    coef = -conf
                else:
                    coef = scipy.stats.t.ppf((1 + conf) / 2, len(intensities) - 1)
                if options.verbose:
                    print('INFO: standard deviation coefficient:', coef)
                self.lb = self.intensity - coef*stds
                self.ub = self.intensity + coef*stds
            else:
                if options.verbose:
                    print('INFO: calculating asymmetric error bars with confidence:', conf, 'and', alg, 'algorithm.')
                self.lb = np.quantile(intensities, (1-conf)/2, axis=0)
                self.ub = np.quantile(intensities, (1+conf)/2, axis=0)
                if alg == 'subsample':
                    print('WARNING: asymmetric error bars with subsampling are experimental.')
                    self.lb = self.intensity - (self.intensity - self.lb) / math.sqrt(nblocks)
                    self.ub = self.intensity + (self.ub - self.intensity) / math.sqrt(nblocks)                    
        return True

    # TODO: remove date from name
    def get_name(self):
        return 'absspec.' + self.infile.split(".")[0] + '.n' + str(self.nsamples) + '.' + self.time.strftime('%Y-%m-%d_%H-%M-%S') # + '.' + str(self.pid)
                    
    def write_spectrum(self, xunit, yunit, index=None):
        indexstr = ''
        if index is not None:
            indexstr = '.' + str(index)
        outfile = self.get_name() + indexstr + "." + xunit[0] + "." + yunit[0] + ".dat"
        print("\n\tPrinting spectrum in units [ " + xunit[2] + ", " + yunit[2]+" ] to " + outfile)
        
        if xunit[0] == "nm":
            x = xunit[1]/self.energies
        else:
            x = self.energies*xunit[1]
        if self.normalize:
            yunit[1] = 1/np.max(self.intensity)
        spectrum = np.hstack((x[:, np.newaxis], yunit[1]*self.intensity[:, np.newaxis]))
        if self.lb is not None and self.ub is not None:
            spectrum = np.hstack((spectrum, yunit[1]*self.lb[:, np.newaxis], yunit[1]*self.ub[:, np.newaxis]))
        if self.intensities is not None:
            spectrum = np.hstack((spectrum, yunit[1]*self.intensities))
        np.savetxt(outfile, spectrum)              

    def writeout(self, index=None):
        xunits = []
        xunits.append(['nm', 1239.8, 'nm'])
        xunits.append(['ev', 1.0, 'eV'])
        xunits.append(['cm', 8065.54, 'cm^-1'])
        yunits = []
        yunits.append(['cross', 1.0, 'cm^2*molecule^-1'])
        yunits.append(['molar', 6.022140e20 / math.log(10), 'dm^3*mol^-1*cm^-1'])
        for xunit in xunits:
            if self.notrans or self.normalize:
                self.write_spectrum(xunit, ['arb', 1.0, 'arb.u.'], index)
                continue
            for yunit in yunits:
                self.write_spectrum(xunit, yunit, index)

class SpectrumBroad(Spectrum):
    """Derived class for spectra with empirical gaussian and/or lorentzian broadening"""

    def __init__(self, nsamples, nstates, deltaE, normalize, notrans, ncores, verbose, minE, maxE, decompose, 
                 sigma, onesigma, sigmaalg, tau):
        super().__init__(nsamples, nstates, deltaE, normalize, notrans, ncores, verbose, minE, maxE, decompose)
        self.sigma = sigma
        self.onesigma = onesigma
        self.sigmaalg = sigmaalg
        self.sigmas = None
        self.tau = tau

        self.dE = None
        self.dist = None
        self.kernel = None
        # self.kernel2 = None # only for sqrtn errorbars
        self.taukernel = None

    def set_sigma(self, samples, weights=None):
        if self.sigmaalg == 'silverman':
            h = silverman(samples, weights)
            if h==0:
                print('WARNING: zero bandwidth from automatic setting, recalculating without weights.')
                print('Energies', samples)
                print('Weights', weights)
                h = silverman(samples)
                print('New bandwidth:', h)
                if h==0:
                    print('WARNING: zero bandwidth even without weights => all energies are the same, setting bandwidth to 1e-6')
                    h = 1.0e-6
        elif self.sigmaalg == 'cv':
#           indices = np.arange(samples.size)
#           np.random.shuffle(indices)
#           samples = samples[indices]
#           weights = weights[indices]
            h = cv(samples, weights, n_jobs=self.ncores)
        return h
            
    def set_sigmas(self, samples=None):
        if self.sigma > 0:
            self.sigmas = self.sigma
            return        
        weights = None
        if samples is None:
            if not self.notrans:
                weights = self.acs
            samples = self.exc
            nsamples = self.exc.shape[0]
        else:
            nsamples = len(samples)
            if not self.notrans:
                weights = self.acs[samples]
            samples = self.exc[samples]          
        if self.sigmaalg == 'dev':
            import kde5
            samples = np.reshape(samples, -1)
            if not self.notrans:
                weights = np.reshape(weights, -1)
            clusters, self.sigmas = kde5.clusterPDE(samples, weights)
            acs = np.transpose(clusters)
            self.acs = np.reshape(acs, (nsamples, self.nstates, len(self.sigmas)))
            return
        if self.onesigma:
            samples = np.reshape(samples, -1)
            if not self.notrans:
                weights = np.reshape(weights, -1)
            self.sigmas = self.set_sigma(samples, weights)
            return
        self.sigmas = np.empty((self.nstates))
        for state in range(self.nstates):
            if self.notrans:
                self.sigmas[state] = self.set_sigma(samples[:, state])
            else:
                self.sigmas[state] = self.set_sigma(samples[:, state], weights[:, state])
                
    def prepare_kernel(self):
        dE = np.subtract.outer(self.energies, self.exc)
        if self.sigma >= 0 or self.sigmas is not None:
            dist = np.power(dE, 2)
            np.divide(dist, -2, out=dist)
            np.exp(dist, out=dist)
            self.dist = dist
        if self.tau > 0.0:
            self.dE = dE
            
    def acs2kernel(self, samples=None):
        if self.sigma >= 0 and self.sigmas is None:
            self.set_sigmas(samples)
            if self.verbose:
                print('sigmas', self.sigmas)
        acs = self.acs
        if samples is not None:
            acs = acs[samples]
        if self.sigmas is not None:
            int_sigma = acs/(self.sigmas*math.sqrt(2*math.pi))
        if self.tau > 0:
            int_tau = acs*(self.tau/2/math.pi)
        if self.sigmas is not None and self.tau > 0: # TODO: move it to kernel2intensity
            np.multiply(int_tau, 1/2, out=int_tau)
            np.multiply(int_sigma, 1/2, out=int_sigma)
    
        if self.sigmas is not None:
            dist = self.dist
            if samples is not None:
                dist = dist[:, samples]
            if self.sigmaalg == 'dev':
                self.kernel = np.power.outer(dist, 1/(self.sigmas**2))
            else:
                self.kernel = np.power(dist, 1/(self.sigmas**2))
            # kernel2 has to be normalized, kernel1 is normalized in int_sigma multipliciation
            # self.kernel2 = self.kernel/self.sigmas/math.sqrt(2*math.pi) # only for sqrtn errorbars
            np.multiply(self.kernel, int_sigma, out = self.kernel)
      
        if self.tau > 0.0:
            self.taukernel = np.divide(int_tau, self.dE**2 + (self.tau**2)/4)
            
    def kernel2intensity(self, samples=None):
        # TODO: unify with kernel2states
        intensity = np.zeros((len(self.energies)))
        if samples is None:
            samples = slice(None)
        if self.tau > 0.0:
            kernel = self.taukernel[:, samples, :]
            np.add(intensity, np.sum(kernel, axis=(1, 2))/kernel.shape[1], out = intensity)
        if self.sigmaalg == 'dev':
            kernel = self.kernel[:, samples, :, :]
            np.add(intensity, np.sum(kernel, axis=(1, 2, 3))/kernel.shape[1], out = intensity)
        elif self.sigmas is not None:
            kernel = self.kernel[:, samples, :]
            np.add(intensity, np.sum(kernel, axis=(1, 2))/kernel.shape[1], out = intensity)
        return intensity
    
    def kernel2states(self, samples):
        intensities = np.zeros((len(self.energies), self.nstates))
        if samples is None:
            samples = slice(None)
        if self.tau > 0.0:
            kernel = self.taukernel[:, samples, :]
            np.add(intensities, np.sum(kernel, axis=(1))/kernel.shape[1], out = intensities)
        if self.sigmaalg == 'dev':
            kernel = self.kernel[:, samples, :, :]
            np.add(intensities, np.sum(kernel, axis=(1, 3))/kernel.shape[1], out = intensities)
        elif self.sigmas is not None:
            kernel = self.kernel[:, samples, :]
            np.add(intensities, np.sum(kernel, axis=(1))/kernel.shape[1], out = intensities)
        return intensities
    
    def kernel2npoints(self, samples):
        npoints = 1e-6*np.ones((len(self.energies)))
        if samples is None:
            samples = slice(None)
        if self.tau > 0.0:
            pass
        if self.sigmaalg == 'dev':
            np.add(npoints, np.sum(self.kernel2[:, samples, :, :], axis=(1, 2, 3)), out = npoints)
        elif self.sigmas is not None:
            np.add(npoints, np.sum(self.kernel2[:, samples, :], axis=(1, 2)), out = npoints)
        return npoints

    def calc_spectrum(self, samples=None):
        self.calc_energies()
        self.trans2acs(normalize=self.notrans)
        self.prepare_kernel()
        self.acs2kernel(samples)
        self.intensity = self.kernel2intensity(samples=samples)
        if self.decompose:
            self.intensities = self.kernel2states(samples)
        # self.npoints = self.kernel2npoints(samples) # only for sqrtn errorbars
        return self.intensity
    
    def recalc_kernel(self, samples=None, clear_sigmas=True):
        # TODO: incorporate to recalc_spectrum
        if clear_sigmas:
            self.sigmas=None
        self.acs2kernel(samples)
        return self.recalc_spectrum()

    def recalc_spectrum(self, samples=None):
        self.intensity = self.kernel2intensity(samples)
        return self.intensity

if __name__ == "__main__":
    start_time = time.time()
    options = read_cmd()
    if options.verbose:
        print("OPTIONS:")
        for option in vars(options):
            print(option, getattr(options, option))
        print()
        print("Number of CPUs on this machine:", cpu_count())
    if options.tau > 0.0 or (options.sigma is not None and options.sigma >= 0):
        spectrum = SpectrumBroad(options.nsamples, options.nstates, options.de, options.normalize, options.notrans,
                                 options.ncores, options.verbose, options.mine, options.maxe,
                                 options.decompose, options.sigma, options.onesigma, options.sigmaalg, options.tau)
    else:
        spectrum = Spectrum(options.nsamples, options.nstates, options.de, options.normalize, options.notrans,
                            options.ncores, options.verbose, options.mine, options.maxe,
                            options.decompose)

    spectrum.read_data(options.infile)
    if options.verbose:
        print('INFO: wall time before calc_spectrum', round(time.time()-start_time), 's')
    spectrum.calc_spectrum()
    if options.ebars:
        if options.verbose:
            print('INFO: wall time before errorbars calculation', round(time.time()-start_time), 's')
        spectrum.calc_errorbars(options.ebars, options.eassym, options.ealg)
    if options.verbose:
        print('INFO: wall time before writeout', round(time.time()-start_time), 's')
    spectrum.writeout()
    if options.verbose:
        print('INFO: wall time', round(time.time()-start_time), 's')
