# -*- coding: utf-8 -*-
"""
Program for the selection of the most representative molecular geometries for spectra modelling.

@author: Stepan Srsen
"""

import numpy as np
import random
import math
import time
import os
import sys
from joblib import Parallel, delayed, cpu_count
from argparse import ArgumentParser
from calc_spectrum_v2 import SpectrumBroad

def read_cmd():
    """Function for command line parsing."""
#    parser = calc_spectrum.read_cmd(parse=False)
    parser = ArgumentParser(description='Spectrum reduction.')
    parser.add_argument('infile', help='Input file.')
    parser.add_argument('-n', '--nsamples', type=int, default=1,
                        help='Number of samples.')
    parser.add_argument('-N', '--nstates', type=int, default=1,
                        help='Number of excited states (ground state not included).')
    parser.add_argument('-d', '--de', type=float, default=0.02,
                        help='Bin step in eV. Default = 0.02 ')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Activate verbose mode.')
    parser.add_argument('--mine', type=float, default=0.0,
                        help='Minimal energy of the spectrum in eV. Default = 0.0')
    parser.add_argument('--maxe', type=float, default=0.0,
                        help='Maximal energy of the spectrum in eV. Default = 0.0')
    parser.add_argument('--normalize', action="store_true", default=False,
                        help='Normalize maximum to one for printed spectra.')
    parser.add_argument('--notrans', action="store_true", default=False,
                        help='No transition dipole moments. Spectrum will be normalized to unity. Useful for ionizations.')
    parser.add_argument('-j', '--ncores', type=int, default=1,
                        help='Number of cores for parallel execution of computatinally intensive subtasks:'
                        + ' cross-validation bandwidth setting, error bars, geometry reduction.')
    
    parser.add_argument('-S', '--subset', type=int, default=0,
                        help='Number of representative molecules.')
    parser.add_argument('-c', '--cycles', type=int, default=1000,
                        help='Number of cycles for geometries reduction.')
    parser.add_argument('-J', '--njobs', dest='njobs', type=int, default=1,
                        help='Number of reduction jobs.')
    parser.add_argument('--pdfcomp', choices=['KLdiv','JSdiv','KStest', 'kuiper', 'SAE', 'RSS', 'cSAE', 'cRSS'], default='KLdiv',
                        help='Method for comparison of probability density functions.')
    
    return parser.parse_args()

class PDFDiv:
    """Class with different methods to calculate the divergence of two probability density functions."""

    @staticmethod
    def KLdiv(pdf1, pdf2, normalized=False, normalize=False):
        """Generalized Kullback-Leibler divergence. pdf1 is used for probabilities."""
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Interpretations
        # maybe normalize both by pdf1 for exact but comparable results
        if normalize or not normalized:
            norm1 = np.sum(pdf1)
            norm2 = np.sum(pdf2)
        if normalize:
            pdf1 /= norm1
            pdf2 /= norm2
            normalized = True
        thr = 1e-15
        if not normalized:
            thr *= norm1
        indices = pdf1>thr
        pdf1 = pdf1[indices]
        pdf2 = pdf2[indices]
        pdf1 = pdf1 + thr
        pdf2 = pdf2 + thr
        
        d = np.divide(pdf1,pdf2)
        np.log(d, out=d)
        np.multiply(d, pdf1, out=d)
        d = np.sum(d)
        if not normalized:
            d += -norm1 + norm2
        return d

    @staticmethod
    def JSdiv(pdf1, pdf2):
        """Jensen–Shannon divergence."""
        pdf3 = (pdf1 + pdf2) / 2
        d = 0.5*PDFDiv.KLdiv(pdf1, pdf3) + 0.5*PDFDiv.KLdiv(pdf2, pdf3)
    #    print(d)
        return d
    
    @staticmethod
    def KStest(pdf1, pdf2):
        """Kolmogorov–Smirnov test."""
        cdf1 = 0.0
        cdf2 = 0.0
        d = 0.0
        for i in range(len(pdf1)):
            cdf1 += pdf1[i]
            cdf2 += pdf2[i]
            dact = abs(cdf1-cdf2)
            if dact > d:
                d = dact
        return d
    
    @staticmethod
    def kuiper(pdf1, pdf2):
        """Kuiper test."""
        cdf1 = 0.0
        cdf2 = 0.0
        dminus = 0.0
        dplus = 0.0
        for i in range(len(pdf1)):
            cdf1 += pdf1[i]
            cdf2 += pdf2[i]
            dminusact = cdf1-cdf2
            dplusact = -dminusact
            if dminusact > dminus:
                dminus = dminusact
            if dplusact > dplus:
                dplus = dplusact
        d = dplus+dminus
        return d
    
    @staticmethod
    def SAE(pdf1, pdf2):
        """Sum of absolute errors/differences."""
        # proc ne suma ctvercu odchylek?
        d = np.sum(np.abs(pdf1-pdf2))
        return d
    
    @staticmethod
    def RSS(pdf1, pdf2):
        """Residual sum of squares."""
        d = np.sum(np.power(pdf1-pdf2, 2))
        return d
    
    @staticmethod
    def cSAE(pdf1, pdf2):
        """Sum of absolute errors/differences of CDFs corresponding to given PDFs."""
        cdf1 = np.cumsum(pdf1)
        cdf2 = np.cumsum(pdf2)
        d = np.sum(np.abs(cdf1-cdf2))
        return d
    
    @staticmethod
    def cRSS(pdf1, pdf2):
        """Residual sum of squares of CDFs corresponding to given PDFs."""
        cdf1 = np.cumsum(pdf1)
        cdf2 = np.cumsum(pdf2)
        d = np.sum(np.power(cdf1-cdf2, 2))
        return d

class GeomReduction:
    def __init__(self, spectrum, nsamples, subset, cycles, ncores, njobs, verbose, pdfcomp, recalc_sigma):
        self.spectrum = spectrum
        self.nsamples = nsamples
        self.subset = subset
        self.cycles = cycles
        self.ncores = ncores
        self.njobs = njobs
        self.verbose = verbose
        self.subsamples = []
        self.origintensity = None
        self.calc_diff = getattr(PDFDiv, pdfcomp)
        self.recalc_sigma = recalc_sigma
        if self.subset==1:
            self.recalc_sigma = False

    def select_subset(self):
        samples = random.sample(range(self.nsamples), self.subset)
        rest = list(set(range(self.nsamples)) - set(samples))
        return samples, rest

    def swap_samples(self, array1, array2):
        index1 = random.randrange(len(array1))
        index2 = random.randrange(len(array2))
        array1[index1], array2[index2] = array2[index2], array1[index1]

    def SA(self, test=False, pi=0.9, pf=0.1, li=None, lf=None):
        if test:
            subsamples = self.subsamples
            restsamples = list(set(range(self.nsamples)) - set(subsamples))
            it = 1
            diffmax = 0
            diffmin = np.inf
        else:
            subsamples, restsamples = self.select_subset()
            subsamples_best = subsamples
            d_best = np.inf
            
            nn = self.subset*(self.nsamples-self.subset)
            if not li:
                itmin = 1
            else:
                itmin = nn*li
            if not lf:
                itmax = int(math.ceil(nn/self.nsamples))
            else:
                itmax = nn*lf
            if itmin==itmax:
                itc = 1
                loops = itmin*self.cycles
            else:
                itc = math.exp((math.log(itmax)-math.log(itmin))/self.cycles)
                loops = int(itmin*(itc**(self.cycles)-1)/(itc-1)) # neglects rounding
            it = itmin
            
            self.subsamples = subsamples[:]
            sa_test_start = time.time()
            ti, tf = self.SA(test=True, pi=pi, pf=pf)
            sa_test_time = time.time() - sa_test_start
            tc = math.exp((math.log(tf)-math.log(ti))/self.cycles)
            temp = ti

        if self.recalc_sigma:
            intensity = self.spectrum.recalc_kernel(samples=subsamples)
        else:
            intensity = self.spectrum.recalc_spectrum(samples=subsamples)
        d = self.calc_diff(self.origintensity, intensity)
        
        if not test:
            m, s = divmod(int(round(sa_test_time*loops/self.cycles)), 60)
            h, m = divmod(m, 60)
            print('Ti', ti, 'Tf', tf)
            print('Li', itmin, 'Lf', itmax)
            toprint = str(self.spectrum.pid)+":\tInitial temperature = "+str(ti)
            toprint += ", Final temperature = "+str(tf)+", Temperature coefficient = "+str(tc)
            toprint += "\n\tMarkov Chain Length coefficient = "+str(itc)+", Initial D-min = "+str(d)
            toprint += "\n\tEstimated run time: "+str(h)+" hours "+str(m)+" minutes "+str(s)+" seconds"
            print(toprint)
#         sys.stdout.flush()

        for _ in range(self.cycles):
            for _ in range(int(round(it))):
                subsamples_i = subsamples[:]
                restsamples_i = restsamples[:]
                self.swap_samples(subsamples_i, restsamples_i)
                if self.recalc_sigma:
                    intensity = self.spectrum.recalc_kernel(samples=subsamples_i)
                else:
                    intensity = self.spectrum.recalc_spectrum(samples=subsamples_i)
                d_i = self.calc_diff(self.origintensity, intensity)
                if test:
                    prob = 1
                    diff = abs(d_i - d)
                    if diff > diffmax:
                        diffmax = diff
                    elif diff < diffmin and diff > 0:
                        diffmin = diff
                else:
                    if d_i < d:
                        prob = 1.0
                        if d_i < d_best:
                            subsamples_best = subsamples_i
                            d_best = d_i
                    else:
                        prob = math.exp((d - d_i)/ temp)
                if prob >= random.random():
                    subsamples = subsamples_i
                    restsamples = restsamples_i
                    d = d_i
            if not test:
                temp *= tc
                it *= itc
        if test:
            print('diffmax', diffmax, 'diffmin', diffmin, 'd', d)
            return -diffmax/math.log(pi), -diffmin/math.log(pf)
        
        if self.recalc_sigma:
            self.spectrum.recalc_kernel(samples=subsamples_best)
        else:
            self.spectrum.recalc_spectrum(samples=subsamples_best)
        self.subsamples = subsamples_best
        return d_best

    def random_search(self):
        div = np.inf
        for i in range(self.cycles):
            subsamples, _ = self.select_subset()
            if self.recalc_sigma:
                intensity = self.spectrum.recalc_kernel(samples=subsamples)
            else:
                intensity = self.spectrum.recalc_spectrum(samples=subsamples)
            div_act = self.calc_diff(self.origintensity, intensity)
            if div_act <= div:
                self.subsamples = subsamples
                div = div_act
                print("Sample"+str(i)+": D-min ="+str(div))
        if self.recalc_sigma:
            self.spectrum.recalc_kernel(samples=self.subsamples)
        else:
            self.spectrum.recalc_spectrum(samples=self.subsamples)
        return div
    
    def extensive_search(self, i):
        self.subsamples = [i]
        # if self.recalc_sigma:
        #     self.spectrum.recalc_kernel(samples=self.subsamples)
        intensity = self.spectrum.recalc_spectrum(samples=self.subsamples)
        div = self.calc_diff(self.origintensity, intensity)
        return div

    def reduce_geoms_worker(self, i, li=None, lf=None):
        name = self.spectrum.get_name() + '.r' + str(self.subset)
        os.chdir(name)
        orig_stdout = sys.stdout
        with open('output.txt', 'a') as f:
           sys.stdout = f
           div = self.SA(li=li, lf=lf)
           self.spectrum.writeout(i)
           self.writegeoms(i)
        sys.stdout = orig_stdout   
        os.chdir('..')
        return div, self.subsamples
    
    def random_geoms_worker(self, i):
        name = self.spectrum.get_name() + '.r' + str(self.subset)
        os.chdir(name)
        orig_stdout = sys.stdout
        with open('output_rnd.txt', 'a') as f:
           sys.stdout = f
           div = self.random_search()
           self.spectrum.writeout("rnd."+str(i))
           self.writegeoms("rnd."+str(i))
        sys.stdout = orig_stdout   
        os.chdir('..')
        return div, self.subsamples
    
    def extensive_search_worker(self, i):
        name = self.spectrum.get_name() + '.r' + str(self.subset)
        os.chdir(name)
        orig_stdout = sys.stdout
        with open('output_ext.txt', 'a') as f:
           sys.stdout = f
           div = self.extensive_search(i)
           self.spectrum.writeout("ext."+str(i))
           self.writegeoms("ext."+str(i))
        sys.stdout = orig_stdout   
        os.chdir('..')
        return div, self.subsamples

    def process_results(self, divs, subsamples, suffix=''):
        print('average divergence', np.average(divs))
        print('divergence std', np.std(divs))
        min_index = np.argmin(divs)
        min_div = divs[min_index]
        self.subsamples = subsamples[min_index]
        print('minimum divergence:', min_div, ', minimum index:', min_index)
        if self.recalc_sigma:
            self.spectrum.recalc_kernel(samples=self.subsamples)
        else:
            self.spectrum.recalc_spectrum(samples=self.subsamples)
        self.spectrum.writeout('r'+str(self.subset)+'.'+suffix+str(min_index))
        self.writegeoms('r'+str(self.subset)+'.'+suffix+str(min_index))


    def reduce_geoms(self):
       # check np.copy vs [:] !
        self.origintensity = np.copy(self.spectrum.calc_spectrum())
        print("Original spectrum sigmas: "+str(self.spectrum.sigmas))
        print("\nPrinting original spectra:")
        self.spectrum.writeout()
        # if not recalc_sigma, the selected geometries do not have to encode peak widths
        # which might be problem for the higher-level method
        if not self.recalc_sigma:
            norm = np.sum(self.spectrum.acs)**2/np.sum(self.spectrum.acs**2)
            # norm = self.nsamples
            self.spectrum.sigmas /= (self.subset/norm)**(1/5)     
            self.spectrum.recalc_kernel(clear_sigmas=False)
            print("Reduced spectra sigmas: "+str(self.spectrum.sigmas))

        name = self.spectrum.get_name() + '.r' + str(self.subset)
        os.mkdir(name)
        
        with Parallel(n_jobs=self.ncores, verbose=1*int(self.verbose)) as parallel:
            divs, subsamples = zip(*parallel(delayed(self.reduce_geoms_worker)(i) for i in range(self.njobs)))
        print('SA divergences:')
        self.process_results(divs, subsamples)
        
        nn = self.subset*(self.nsamples-self.subset)
        itmin = 1
        itmax = int(math.ceil(nn/self.nsamples))
        itc = math.exp((math.log(itmax)-math.log(itmin))/self.cycles)
	# calculate # of loops to provide comparable resources to random search
        loops=0
        it=itmin
        for _ in range(self.cycles):
            for _ in range(int(round(it))):
                loops+=1
            it*=itc
        print('# of loops', loops)
        # print('loops approx.', int(itmin*(itc**(self.cycles)-1)/(itc-1)), 'Li', itmin, 'Lm', itmax)
        self.cycles = loops
        with Parallel(n_jobs=self.ncores, verbose=1*int(self.verbose)) as parallel:
            divs, subsamples = zip(*parallel(delayed(self.random_geoms_worker)(i) for i in range(self.njobs)))
        print('Random divergences:')
        self.process_results(divs, subsamples, suffix='rnd.')
        
        if self.subset==1:
            with Parallel(n_jobs=self.ncores, verbose=1*int(self.verbose)) as parallel:
                divs, subsamples = zip(*parallel(delayed(self.extensive_search_worker)(i) for i in range(self.nsamples)))
            print('Extensive search = global minimum:')
            self.process_results(divs, subsamples, suffix='ext.')

    def writegeoms(self, index=None):
        indexstr = ''
        if index is not None:
            indexstr = '.' + str(index)
        outfile = self.spectrum.get_name() + indexstr + '.geoms'
#      print(str(self.spectrum.pid)+":\tPrinting geometries of reduced spectrum to "+outfile)
        with open(outfile, "w") as f:
            for i in self.subsamples:
    #         f.write('%s' % (self.samples[i]))
                f.write('%s\n' % (i+1))
                
if __name__ == "__main__":
    random.seed(0)
    start_time = time.time()
    options = read_cmd()
    if options.verbose:
        print("OPTIONS:")
        for option in vars(options):
            print(option, getattr(options, option))
        print()
        print("Number of CPUs on this machine:", cpu_count())
    spectrum = SpectrumBroad(options.nsamples, options.nstates, options.de, options.normalize, options.notrans,
                                 options.ncores, options.verbose, options.mine, options.maxe,
                                 decompose=False, sigma=0, onesigma=False, sigmaalg='silverman', tau=0)
    spectrum.read_data(options.infile)
    # os.chdir('DATA')
    geomReduction = GeomReduction(spectrum, options.nsamples, options.subset, options.cycles,
                                  options.ncores, options.njobs, options.verbose, options.pdfcomp,
                                  recalc_sigma=True)
    geomReduction.reduce_geoms()
    
    if options.verbose:
        print('INFO: wall time', round(time.time()-start_time), 's')
