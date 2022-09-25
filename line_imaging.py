import numpy as np
import matplotlib.pyplot as plt
from jwst import datamodels
from scipy.signal import medfilt
from matplotlib.colors import LogNorm
from astropy.modeling import models
from astropy.io import fits

def FindNearest(array, value):
    '''
    Find index of item in array closest to a given value
    
        Parameters:
            array (array-like): Array to find the index of
            value (float): Value to compare to items in the array
            
        Returns:
            idx (int): index of item in array closest to value
    
    '''
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
    
def ReduceCube(cube_file, med_kernel):
    '''
    Load in and continuum subtract an IFU datacube
    
        Parameters:
            cube_file(str): Path to the cube file (s3d.fits file)
            med_kernel (odd int): Kernal for the median filter used to estimate the continuum
            
        Returens:
            cont_sub_cube (numpy.ndarray): Continuum subracted IFU data cube
            cont_cube (numpy.ndarray): Estimate of continuum data cube
    
    '''
    
    # load in data into datamodels format and extract data
    data_cube = datamodels.IFUCubeModel(cube_file).data
    n_chan, dim1, dim2 = data_cube.shape
    
    # use a median filter to estimate the continuum for each pixel
    # subtract pixel-wise to get a continuum subracted cube
    cont_cube = np.zeros((n_chan, dim1, dim2))
    cont_sub_cube = np.zeros((n_chan, dim1, dim2))
    
    for i in range(dim1):
        for j in range(dim2):
            raw_spec = data_cube[:, i, j] # spectrum at pixel i, j
            cont = medfilt(raw_spec, med_kernel) # use median filter to estimate continuum
            cont_cube[:, i, j] = cont
            cont_sub_cube[:, i, j] = raw_spec - cont
            
    return cont_sub_cube, cont_cube
    
def ExtractSpec(data_cube, rad):
    '''
    Extract 1D spectra from IFU data cube by taking the median of circular region in each channel
    
        Parameters:
            data_cube (numpy.ndarray): Continuum subtracted data cube
            rad (int): Radius of circular region to take median of in spectra extraction
            
        Returns:
            ext_spec (numpy.ndarray): Extracted 1D specrum
    '''
    
    n_chan, dim1, dim2 = data_cube.shape
    # find index of middel pixel in each dimension of the image
    mid_idx1, mid_idx2 = (dim1 - 1) / 2, (dim2 - 1) / 2
    
    # shrink cube to be size nchan, rad, rad
    left_edge, right_edge = int(mid_idx2 - rad), int(mid_idx2 + rad + 1)
    lower_edge, upper_edge = int(mid_idx1 - rad), int(mid_idx1 + rad + 1)
    
    small_cube = data_cube[:, lower_edge:upper_edge, left_edge:right_edge]
    # redfine dimensions and middle index
    n_chan, dim1, dim2 = small_cube.shape
    mid_idx1, mid_idx2 = (dim1 - 1) / 2, (dim2 - 1) / 2
    
    ext_spec = np.zeros(n_chan)
    
    # take median of pixels in central circle of each channel
    for i in range(n_chan):
        fluxes = []
        img = small_cube[i]
        for y in range(dim1):
            for x in range(dim2):
                if (x - mid_idx2)**2 + (y - mid_idx1)**2 < rad**2:
                    fluxes.append(img[y, x])
        ext_spec[i] = np.median(fluxes)
        
    return ext_spec
    

def GetWvl(spec_file):
    '''
    Get wavelengths from extracted spectrum file
    
        Parameters:
            spec_file (str): Path to extracted specturm (x1d.fits) file
            
        Returns:
            wvl (numpy.ndarray): wavelength of each channel in the IFU data cube [um]
    '''
    spec_model = datamodels.SpecModel(spec_file)
    return spec_model.spec_table['WAVELENGTH']
    
    
def CheckLines(spec, line_dict, lines, wvl, stat_window, line_width, noise_thresh):
    '''
    Check if the lines in given line lists are present in spectrum. Checks if peaks are a certian threshold above the noise.
    
        Parameters:
            spec (numpy.ndarrays): Input 1D spectrum
            line_dict (dict): dictionary of line lists (name: line list)
            lines (list): list of lines to check for in spectrum, lists stored in dictionary
            wvl (numpy.ndarray): wavelenghts of each channel in spectrum
            stat_window (int): size of window used to estimate the noise (sigma)
            line_width (int): size estimate of spectral lines in pixels
            thresh (float): how many multiples greater than sigma a peak must be to be included
            
        Returns:
            found_lines (list of array-likes): list of updated lines lists with only found lines
    '''
    # reduce line list to be within the wavelegnth range of the spectrum
    n_lines = len(lines)
    reduced_lines = {}
    for i in range(n_lines):
        line_list = line_dict[lines[i]]
        idxs1 = np.where(line_list > np.min(wvl))[0]
        idxs2 = np.where(line_list < np.max(wvl))[0]
        idxs = [val for val in idxs1 if val in idxs2] # find indicies between min and max of wvl
        reduced_lines[lines[i]] = line_list[idxs]
    # create a peakless spectrum by interpolating out peaks
    peakless_spec = np.copy(spec)
    for line in lines:
        line_list = reduced_lines[line]
        for i in range(len(line_list)):
            peak = line_list[i]
            peak_idx = FindNearest(wvl ,peak) # index of wvl closest to peak
            left_idx, right_idx = peak_idx - line_width, peak_idx + line_width
            
            # use linear interpolation to get rid of peak
            x1, x2 = wvl[left_idx], wvl[right_idx]
            y1, y2 = spec[left_idx], spec[right_idx]
            line_idxs = range(left_idx, right_idx + 1) # indicies of the current line
            
            for j in line_idxs:
                # perform linear interpolation
                peakless_spec[j] = y1 + (wvl[j] - x1) * ((y2-y1) / (x2-x1))

    found_lines = {}
    for line in lines:
        found_lines[line] = []
    for i in range(n_lines):
        line_list = reduced_lines[lines[i]]
        for j in range(len(line_list)):
            peak = line_list[j]
            peak_idx = FindNearest(wvl, peak)
            peak_val = spec[peak_idx]
            # estimate sigma by taking std of region around line in the peakless spectrum
            window = peakless_spec[peak_idx - stat_window: peak_idx + stat_window + 1]
            sigma = np.std(window)
            if peak_val > noise_thresh * sigma:
                found_lines[lines[i]].append(peak)
                
    return found_lines
    
    
def GenPlots(data_cube, spec, wvl, found_lines, output_dir, line_width, win_width, save_fits):
    '''
    Generates line image plots side by side with spectral line
    
        Parameters:
            data_cube (numpy.ndarray): IFU data cube
            spec (numpy.ndarray): Extracted 1D spectrum
            wvl (array-like): wavelength array for spec
            found_lines (dict): dict of line_lists to image (name: lines)
            output_dir (str): directory to save the images
            line_width (int): estimate of spectral line width in pixels
            win_width (int): width of window used in spectral line plot
            
        Returns:
            None
    '''
    lines = []
    for line in found_lines:
        lines.append(line)
    
    n_lines = len(lines)
    for ii in range(n_lines):
        line_list = found_lines[lines[ii]]
        for i in range(len(line_list)):
            idx = FindNearest(wvl, line_list[i])
            # sum all channels in given line
            img_list = data_cube[idx - line_width: idx + line_width + 1]
            img = np.sum(img_list, axis=0)
            
            # save fits file of image
            if save_fits:
                hdu = fits.PrimaryHDU(img)
                hdul = fits.HDUList([hdu])
                hdul.writeto(output_dir + str(lines[ii]) + '_' + str(i) + '.fits')
                
            # create a gaussian representing the resolution:
            fwhm = wvl[idx] / 800 # estimate dlambda/lambda = 800
            sigma = fwhm / (2*np.sqrt(2*np.log(2)))
            g = models.Gaussian1D(amplitude=spec[idx], mean=wvl[idx], stddev=sigma)
            x = np.linspace(wvl[idx-win_width], wvl[idx+win_width], 50)
            y = g(x)
            # create side-by-side plot of line image and spectral line
            plt.close()
            fig, ax = plt.subplots(1, 2)
            # plot image on log scale
            ax[0].imshow(img, origin='lower', norm=LogNorm())
            
            ax[1].plot(wvl[idx-win_width:idx+win_width+1], spec[idx-win_width:idx+win_width+1])
            ax[1].axvline(x=line_list[i], color='black')
            ax[1].axvline(x=wvl[idx - line_width], ls=':')
            ax[1].axvline(x=wvl[idx + line_width], ls=':')
            ax[1].plot(x, y, color='red', ls='--')
            
            fname = str(lines[ii]) + '_' + str(i) + '.png'
            
            plt.savefig(output_dir + fname)

    
    
    
    
    
    
    
    
    
    
