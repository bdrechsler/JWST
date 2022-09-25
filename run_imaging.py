from line_imaging import *
from lines import line_dict

# define paths to data files
data_dir = '/users/bdrechsl/jax_2/bdrechsl/IRAS16253/JWST/jw01802-o015_t012_nirspec_g395m-f290lp/'
cube_file = data_dir + 'jw01802-o015_t012_nirspec_g395m-f290lp_s3d.fits'
spec_file = data_dir + 'jw01802-o015_t012_nirspec_g395m-f290lp_x1d.fits'

output_dir = '/users/bdrechsl/jax_2/bdrechsl/IRAS16253/Plots/'

# define parameters for imaging
med_kernel = 25 # kernel for median filter used to estimate continuum for continuum subtraction
rad = 10 # radius of circular aperture used in spectra extraction
stat_window = 15 # (half) size of window used to estamte sigma when checking if lines are present
line_width = 3 # estmate of (half) line width in pixels
noise_thresh = 3 # threshold above noise for line to be considered present
win_width = 7 # size of window (pixels) for plot of spectral line being imgaged
save_fits = True # determines if line image is saved as a fits file
lines = ['H', 'H2']

# Run analysis and plotting functions
print('Continuum subtracting data cube ...')
data_cube, cont_cube = ReduceCube(cube_file, med_kernel)
print('Extracting 1D spectrum ...')
spec = ExtractSpec(data_cube, rad)
wvl = GetWvl(spec_file)
print('Checking Lines ...')
found_lines = CheckLines(spec, line_dict, lines, wvl, stat_window, line_width, noise_thresh)
print('Plotting ...')
GenPlots(data_cube, spec, wvl, found_lines, output_dir, line_width, win_width, save_fits)
