# -*- coding: utf-8 -*-
"""
Program for the selection of the most representative molecular geometries for spectra modelling.

@author: Stepan Srsen
"""

import sys
import numpy as np
import random
import math
import time
import os
from joblib import Parallel, delayed, cpu_count
from argparse import ArgumentParser
import datetime
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform

def read_cmd():
    """Function for command line parsing."""
#    parser = calc_spectrum.read_cmd(parse=False)
    parser = ArgumentParser(description='Spectrum reduction.')
    parser.add_argument('infile', help='Input file.')
    parser.add_argument('-n', '--nsamples', type=int, default=1,
                        help='Number of samples.')
    parser.add_argument('-N', '--nstates', type=int, default=1,
                        help='Number of excited states (ground state not included).')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Activate verbose mode.')
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
        # print(pdf1.shape)
        pdf1 = pdf1[indices]
        # print(pdf1.shape)
        pdf2 = pdf2[indices]
        pdf1 = pdf1 + thr
        pdf2 = pdf2 + thr
        
        d = np.divide(pdf1,pdf2)
        np.log(d, out=d)
        np.multiply(d, pdf1, out=d)
        d = np.sum(d)
        if not normalized:
            d += -norm1 + norm2
    #    print(d)
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

# def bandwidth(kde):
#     # h = kde.silverman_factor()
#     # print(h)
#     h = 1
#     return h

class GeomReduction:
    def __init__(self, nsamples, nstates, subset, cycles, ncores, njobs, verbose, pdfcomp):
        self.nsamples = nsamples
        # if nstates > 1:
        #     print("ERROR: implemented only for 1 state!")
        #     return False
        self.nstates = nstates
        self.exc = np.empty((nsamples, nstates))
        self.trans = np.empty((nsamples, nstates, 3))
        self.grid = None
        self.subset = subset
        self.cycles = cycles
        self.ncores = ncores
        self.njobs = njobs
        self.verbose = verbose
        self.subsamples = []
        self.sweights = None
        self.origintensity = None
        self.calc_diff = getattr(PDFDiv, pdfcomp)
        self.pid = os.getpid()
            
    def read_data(self, infile):
        self.infile = infile
        self.time = datetime.datetime.now()
        with open(self.infile, "r") as f:
            i = 0 #line
            j = 0 #sample
            k = -1 #state
            for line in f:
                if (i % 2 == 1):
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
                    if j >= self.nsamples:
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
            if (i != 2*self.nsamples*self.nstates):
                print("Error: Number of transitions in the input file is smaller than the number of samples multiplied by the number of states.")
                sys.exit(1)

        self.trans = np.power(self.trans,2)
        self.trans = np.sum(self.trans, axis=2)
        self.weights = self.exc*self.trans
        self.wnorms = np.sum(self.weights, axis=0)/np.sum(self.weights)
        
    def get_name(self):
        return 'absspec.' + self.infile.split(".")[0] + '.n' + str(self.nsamples) + '.' + self.time.strftime('%Y-%m-%d_%H-%M-%S') # + '.' + str(self.pid)
        
    def get_PDF(self, samples=None, sweights=None, h='silverman', weighted=True, gen_grid=False, plot=False):
        # TODO: compare each state separately or create common grid and intensity
        # TODO: weight states by corresponding integral intensity, i.e. sum(ene*trans**2)
        if samples is None:
            samples = slice(None)
            plot = True
        
        if gen_grid:
            self.n_points = 100
            n_sigma = 1
            
            # self.exc_min = np.zeros((self.nstates))
            # self.exc_max = np.zeros((self.nstates))
            # self.trans_min = np.zeros((self.nstates))
            # self.trans_max = np.zeros((self.nstates))
            # self.norm = np.zeros((self.nstates))
            # self.grid = np.zeros((self.nstates, 2, self.n_points**2))
            
            h1 = np.amax(np.std(self.exc[samples], axis=0))
            h2 = np.amax(np.std(self.trans[samples], axis=0))
            self.exc_min = self.exc[samples].min() - n_sigma*h1
            self.exc_max = self.exc[samples].max() + n_sigma*h1
            self.trans_min = self.trans[samples].min() - n_sigma*h2
            self.trans_max = self.trans[samples].max() + n_sigma*h2
            X, Y = np.mgrid[self.exc_min : self.exc_max : self.n_points*1j, self.trans_min : self.trans_max : self.n_points*1j]
            dX = (self.exc_max - self.exc_min)/(self.n_points-1)
            dY = (self.trans_max - self.trans_min)/(self.n_points-1)
            norm = 1
            if weighted:
                if sweights is not None:
                    norm = np.sum(self.weights[samples]*sweights)/np.sum(sweights)
                else:
                    norm = np.sum(self.weights[samples])/len(self.weights[samples])
            self.norm = dX*dY/norm
            self.grid = np.vstack([X.ravel(), Y.ravel()])
            if self.subset == 1:
                self.kernel = []
        
        # pdf = np.zeros((self.nstates, self.n_points**2))
        pdf = np.zeros((self.n_points**2))
            
        for state in range(self.nstates):
            exc = self.exc[samples,state]
            trans = self.trans[samples,state]
            values = np.vstack([exc, trans]) # TODO: index values directly
            # h = bandwidth
            norm = self.wnorms[state]
            weights = None
            if weighted:
                if sweights is not None:
                    norm = np.sum(self.weights[samples,state]*sweights)/np.sum(sweights)
                    weights = self.weights[samples,state]*sweights
                else:
                    norm = np.sum(self.weights[samples,state])/len(self.weights[samples,state])
                    weights = self.weights[samples,state]
            elif sweights is not None:
                weights = sweights
            if gen_grid and self.subset == 1:
                kernel = gaussian_kde(values, bw_method=h, weights=weights)
                self.kernel.append(kernel)
            elif self.subset == 1:
                kernel = self.kernel[state]
                kernel.dataset = values
                norm *= self.nsamples
            else:
                kernel = gaussian_kde(values, bw_method=h, weights=weights)
            
            # if gen_grid:
            #     h1 = kernel.covariance[0,0]**0.5
            #     h2 = kernel.covariance[1,1]**0.5
            #     print('bandwidths state', state, ':', h1, h2)
            #     self.exc_min[state] = exc.min() - n_sigma*h1
            #     self.exc_max[state] = exc.max() + n_sigma*h1
            #     self.trans_min[state] = trans.min() - n_sigma*h2
            #     self.trans_max[state] = trans.max() + n_sigma*h2
            #     X, Y = np.mgrid[self.exc_min[state] : self.exc_max[state] : self.n_points*1j, self.trans_min[state] : self.trans_max[state] : self.n_points*1j]
            #     dX = (self.exc_max[state] - self.exc_min[state])/(self.n_points-1)
            #     dY = (self.trans_max[state] - self.trans_min[state])/(self.n_points-1)
            #     # self.gweights = X.ravel()*Y.ravel()
            #     self.norm[state] = dX*dY/norm
            #     self.grid[state] = np.vstack([X.ravel(), Y.ravel()])
                
            # pdf[state] = kernel(self.grid[state])*self.norm[state]*norm #*self.gweights
            pdf += kernel(self.grid)*self.norm*norm#*self.wnorms[state] #*self.gweights
            # print('pdf sum', np.sum(pdf), norm)
        
        # pdf 
                
        if plot:
            # print('pdf sum', np.sum(pdf[state]))
            print('pdf sum', np.sum(pdf))
        plot=False
        if plot:
            import matplotlib.pyplot as plt
            # Z = np.reshape(pdf[state].T, (self.n_points,self.n_points))
            # plt.figure()
            # plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,extent=[self.exc_min[state], self.exc_max[state], self.trans_min[state], self.trans_max[state]], aspect='auto')
            # plt.plot(exc, trans, 'k.', markersize=2)
            # plt.xlim([self.exc_min[state], self.exc_max[state]])
            # plt.ylim([self.trans_min[state], self.trans_max[state]])
            Z = np.reshape(pdf.T, (self.n_points,self.n_points))
            plt.figure()
            plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,extent=[self.exc_min, self.exc_max, self.trans_min, self.trans_max], aspect='auto')
            plt.plot(self.exc[samples].ravel(), self.trans[samples].ravel(), 'k.', markersize=2)
            plt.xlim([self.exc_min, self.exc_max])
            plt.ylim([self.trans_min, self.trans_max])
            plt.show()
            
        return pdf
        
    def select_subset(self, gen_weights=False, randomly=True):
        if randomly:
            samples = np.array(random.sample(range(self.nsamples), self.subset))
        else:
            if self.nstates > 1:
                print('ERROR: intial subset generation with maximal distances is not supported for multiple states.')
                return
            exc = self.exc[:,0] # only for one state
            trans = self.trans[:,0]
            weights = self.weights[:,0]
            exc = exc/np.average(exc, weights=weights)
            trans = trans/np.average(trans, weights=weights)
            values = np.vstack([exc, trans]).T
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(exc, trans, 'k.', markersize=2)
            dists = squareform(pdist(values))
            samples = [np.argmax(np.sum(dists, axis=1))]
            while len(samples) < self.subset:
                sample = np.argmax(np.min(dists[:,samples], axis=1))
                samples.append(sample)
            samples = np.array(samples)
            # self.get_PDF(samples, plot=True)
        
        if gen_weights and self.subset>1:
            weights = int(self.nsamples/self.subset + 0.5)*np.ones(samples.shape, dtype=int)
        else:
            weights = None
        return samples, weights
    
    def swap_samples(self, samples, weights=None):
        index1 = random.randrange(len(samples))
        change_weights = np.random.randint(5)
        # change_weights = 1
        if change_weights==0 or weights is None:
            rest = list(set(range(self.nsamples)) - set(samples))
            index2 = random.randrange(len(rest))
            samples[index1] = rest[index2]
            return samples, weights
        index2 = random.randrange(len(samples))
        while weights[index2]==1 or index1==index2:
            index1 = random.randrange(len(samples))
            index2 = random.randrange(len(samples))
        weights[index1] += 1
        weights[index2] -= 1
        # add = np.random.randint(2)
        # if add or weights[index1]==1:
        #     weights[index1] += 1
        # else:
        #     weights[index1] -= 1
        return samples, weights

    def SA(self, test=False, pi=0.9, pf=0.1, li=None, lf=None):
        if test:
            subsamples = self.subsamples
            weights = self.sweights
            it = 1
            diffmax = 0
            diffmin = np.inf
        else:
            subsamples, weights = self.select_subset()
            subsamples_best = subsamples
            weights_best = weights
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
            
            self.subsamples = np.copy(subsamples)
            if weights is not None:
                self.sweights = np.copy(weights)
            sa_test_start = time.time()
            ti, tf = self.SA(test=True, pi=pi, pf=pf)
            sa_test_time = time.time() - sa_test_start
            tc = math.exp((math.log(tf)-math.log(ti))/self.cycles)
            temp = ti

        intensity = self.get_PDF(samples=subsamples, sweights=weights)
        d = self.calc_diff(self.origintensity, intensity)
        # d = 0
        # for state in range(self.nstates):
        #     d += self.calc_diff(self.origintensity, intensity)*self.wnorms[state]
        
        if not test:
            m, s = divmod(int(round(sa_test_time*loops/self.cycles)), 60)
            h, m = divmod(m, 60)
            print('Ti', ti, 'Tf', tf)
            print('Li', itmin, 'Lf', itmax)
            toprint = str(self.pid)+":\tInitial temperature = "+str(ti)
            toprint += ", Final temperature = "+str(tf)+", Temperature coefficient = "+str(tc)
            toprint += "\n\tMarkov Chain Length coefficient = "+str(itc)+", Initial D-min = "+str(d)
            toprint += "\n\tEstimated run time: "+str(h)+" hours "+str(m)+" minutes "+str(s)+" seconds"
            print(toprint)
#         sys.stdout.flush()
            
        for _ in range(self.cycles):
            for _ in range(int(round(it))):
                subsamples_i = np.copy(subsamples)
                weights_i = None
                if weights is not None:
                    weights_i = np.copy(weights)
                subsamples_i, weights_i = self.swap_samples(subsamples_i, weights_i)
                intensity = self.get_PDF(samples=subsamples_i, sweights=weights_i)
                d_i = self.calc_diff(self.origintensity, intensity)
                # d_i = 0
                # for state in range(self.nstates):
                #     d_i += self.calc_diff(self.origintensity, intensity)*self.wnorms[state]
                # print('d', d)
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
                            weights_best = weights_i
                            d_best = d_i
                    else:
                        prob = math.exp((d - d_i)/ temp)
                if prob >= random.random():
                    subsamples = subsamples_i
                    weights = weights_i
                    d = d_i
            if not test:
                temp *= tc
                it *= itc
        if test:
            print('diffmax', diffmax, 'diffmin', diffmin, 'd', d)
            return -diffmax/math.log(pi), -diffmin/math.log(pf)
        
        self.get_PDF(subsamples_best, sweights=weights_best, plot=True)
        print('best d', d_best)
        self.subsamples = subsamples_best
        self.sweights = weights_best
        print(subsamples_best, weights_best)
        return d_best

    # def random_search(self):
    #     div = np.inf
    #     for i in range(self.cycles):
    #         subsamples, _ = self.select_subset()
    #         if self.recalc_sigma:
    #             intensity = self.spectrum.recalc_kernel(samples=subsamples)
    #         else:
    #             intensity = self.spectrum.recalc_spectrum(samples=subsamples)
    #         div_act = self.calc_diff(self.origintensity, intensity)
    #         if div_act <= div:
    #             self.subsamples = subsamples
    #             div = div_act
    #             print("Sample"+str(i)+": D-min ="+str(div))
    #     if self.recalc_sigma:
    #         self.spectrum.recalc_kernel(samples=self.subsamples)
    #     else:
    #         self.spectrum.recalc_spectrum(samples=self.subsamples)
    #     return div
    
    # def extensive_search(self, i):
    #     self.subsamples = [i]
    #     # if self.recalc_sigma:
    #     #     self.spectrum.recalc_kernel(samples=self.subsamples)
    #     intensity = self.spectrum.recalc_spectrum(samples=self.subsamples)
    #     div = self.calc_diff(self.origintensity, intensity)
    #     return div

    def reduce_geoms_worker(self, i, li=None, lf=None):
        div = self.SA(li=li, lf=lf)
        name = self.get_name() + '.r' + str(self.subset)
        os.chdir(name)
        self.writegeoms(i)
        os.chdir('..')
        return div
    
    # def random_geoms_worker(self, i):
    #     div = self.random_search()
    #     name = self.spectrum.get_name() + '.r' + str(self.subset)
    #     os.chdir(name)
    #     self.spectrum.writeout("rnd"+str(i))
    #     self.writegeoms("rnd"+str(i))
    #     os.chdir('..')
    #     return div
    
    # def extensive_search_worker(self, i):
    #     div = self.extensive_search(i)
    #     name = self.spectrum.get_name() + '.r' + str(self.subset)
    #     os.chdir(name)
    #     self.spectrum.writeout("ext"+str(i))
    #     self.writegeoms("ext"+str(i))
    #     os.chdir('..')
    #     return div

    def reduce_geoms(self):
        self.origintensity = self.get_PDF(gen_grid=True)
        if self.subset == 1:
            for kernel in self.kernel:
                kernel.set_bandwidth(bw_method=1)

        name = self.get_name() + '.r' + str(self.subset)
        os.mkdir(name)
        
        with Parallel(n_jobs=self.ncores, verbose=1*int(self.verbose)) as parallel:
            divs = parallel(delayed(self.reduce_geoms_worker)(i) for i in range(self.njobs))
        print('SA divergences:')
        print('average divergence', np.average(divs))
        print('divergence std', np.std(divs))
        min_index = np.argmin(divs)
        print('minimum divergence:', divs[min_index], ', minimum index:', min_index)
        
        # nn = self.subset*(self.nsamples-self.subset)
        # itmin = 1
        # itmax = int(math.ceil(nn/self.nsamples))
        # itc = math.exp((math.log(itmax)-math.log(itmin))/self.cycles)
        # loops=0
        # it=itmin
        # for _ in range(self.cycles):
        #     for _ in range(int(round(it))):
        #         loops+=1
        #     it*=itc
        # print('loops', loops)
        # # print('loops approx.', int(itmin*(itc**(self.cycles)-1)/(itc-1)), 'Li', itmin, 'Lm', itmax)
        # self.cycles = loops
        # with Parallel(n_jobs=self.ncores, verbose=1*int(self.verbose)) as parallel:
        #     divs = parallel(delayed(self.random_geoms_worker)(i) for i in range(self.njobs))
        # print('Random divergences:')
        # print('average divergence', np.average(divs))
        # print('divergence std', np.std(divs))
        # min_index = np.argmin(divs)
        # print('minimum divergence:', divs[min_index], ', minimum index:', min_index)
        
        # if self.subset==1:
        #     with Parallel(n_jobs=self.ncores, verbose=1*int(self.verbose)) as parallel:
        #         divs = parallel(delayed(self.extensive_search_worker)(i) for i in range(self.nsamples))
        #     min_index = np.argmin(divs)
        #     print('Global minimum divergence:', divs[min_index], ', minimum index:', min_index)

    def writegeoms(self, index=None):
        indexstr = ''
        if index is not None:
            indexstr = '.' + str(index)
        outfile = self.get_name() + indexstr + '.geoms'
        with open(outfile, "w") as f:
            for i in range(len(self.subsamples)):
                if self.sweights is None:
                    f.write('%s\n' % (self.subsamples[i]+1))
                else:
                    f.write('%s %s\n' % (self.subsamples[i]+1, self.sweights[i]))
                
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

    geomReduction = GeomReduction(options.nsamples, options.nstates, options.subset, options.cycles,
                                  options.ncores, options.njobs, options.verbose, options.pdfcomp)
    geomReduction.read_data(options.infile)
    geomReduction.reduce_geoms()
    
    if options.verbose:
        print('INFO: wall time', round(time.time()-start_time), 's')
