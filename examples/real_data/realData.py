# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]

## TIME SLICE QUERY 
## num antennas, num stations????
# #############################################################################

"""
Simulation LOFAR imaging with Bipp (NUFFT).
"""
import sys
from tqdm import tqdm as ProgressBar
import astropy.units as u
import astropy.coordinates as coord
import astropy.io.fits as fits
import astropy.wcs as awcs
import astropy.time as atime
import bipp.imot_tools.io.s2image as s2image
import numpy as np
import scipy.constants as constants
import bipp
import bipp.beamforming as beamforming
import bipp.gram as bb_gr
import bipp.parameter_estimator as bb_pe
import bipp.source as source
import bipp.instrument as instrument
import bipp.frame as frame
import bipp.statistics as statistics
import bipp.measurement_set as measurement_set
import time as tt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
# for final figure text sizes
plt.rcParams.update({'font.size': 26})



start_time= tt.time()
#####################################################################################################################################################################
# Input and Control Variables #####################################################################################################################################################################
#####################################################################################################################################################################
try:
    telescope_name =sys.argv[1]
    ms_file = sys.argv[2]
    filter_negative_eigenvalues= True

    if (telescope_name.lower()=="skalow"):
        ms=measurement_set.SKALowMeasurementSet(ms_file)
        N_station = 512
        N_antenna = 512
        

    elif (telescope_name.lower()=="mwa"):
        ms = measurement_set.MwaMeasurementSet(ms_file)
        N_station = 128
        N_antenna = 128

    elif (telescope_name.lower()=="lofar"):
        ms = measurement_set.LofarMeasurementSet(ms_file)
        N_station = 38 # netherlands, 52 international
        N_antenna = 38 # netherlands, 52 international
    else: 
        raise(NotImplementedError("A measurement set class for the telescope you are searching for has not been implemented yet - please feel free to implement the class yourself!"))

except: 
    raise(SyntaxError("This file must be called with the telescope name and path to ms file at a minimum. "+\
                      "Eg:\npython realData.py [telescope name(string)] [path_to_ms(string)] [output_name(string)] [N_pix(int)] [FoV(float(deg))] [N_levels(int)] [Clustering(bool/list_of_eigenvalue_bin_edges)] [WSCleangrid(bool)] [WSCleanPath(string)]"))

try:
    outName=sys.argv[3]
except:
    outName="test"
try:
    N_pix=int(sys.argv[4])
except:
    N_pix=512
try:
    FoV = np.deg2rad(float (sys.argv[5]))
except:
    FoV = np.deg2rad(6) # For lofar + Mwa
"""
 # time and channel id stuff to be implemented!!!
try:
    try:
        channel_id=bool(sys.argv[6])
        if (channel_id==True):
            # put code to read all channels
            channel_id = np.array()
    else:
        channelIDstr=sys.argv[10].split(",")
        channels=[]
        for channelID in channelIDstr:
            channel_id.append(int(channelID))
        channel_id = np.array(channel_id)


try:
    N_level = int(sys.argv[8])
except:
    N_level=3
try:
    try:
        clustering = bool(sys.argv[9]) # True or set of numbers which act as bins,separated by commas and NO spaces
        clusteringBool = True
    except:
        binEdgesStr = sys.argv[10].split(",")
        clustering = []
        for binEdge in binEdgesStr:
            clustering.append(float(binEdge))
        clustering= np.array(clustering)
        clusteringBool = False
"""
try:
    N_level = int(sys.argv[6])
except:
    N_level=3
try:
    try:
        clusterEdges = np.array(sys.argv[7].split(","), dtype=np.float32)
        clusteringBool = False
        binStart = clusterEdges[0]
        clustering = []
        for binEdge in clusterEdges[1:]:
            binEnd = binEdge
            clustering.append([binStart, binEnd])
            binStart = binEnd
        clustering = np.asarray(clustering, dtype=np.float32)
    except:
        clustering = bool(sys.argv[7]) # True or set of numbers which act as bins,separated by commas and NO spaces
        clusteringBool = True
        
except:
    clustering = True
    clusteringBool= True

try:
    partitions=int(sys.argv[8])
except:
    partitions = 1
try:
    WSClean_grid = bool(sys.argv[9])
    if (WSClean_grid == True):
        ms_fieldcenter = False
    else: 
        ms_fieldcenter = True
except:
    WSClean_grid=False
    ms_fieldcenter=True


try:
    wsclean_path = sys.argv[10]
except:
    if (WSClean_grid==True):
        raise(SyntaxError("If WSClean Grid is set to True then path to wsclean fits file must be provided!"))
    else:
        print ("WSClean fits file not provided.")
        wsclean_path= ""



print (f"Telescope Name:{telescope_name}")
print (f"MS file:{ms_file}")
print (f"Output Name:{outName}")
print (f"N_Pix:{N_pix} pixels")
print (f"FoV:{np.rad2deg(FoV)} deg")
print (f"N_level:{N_level} levels")
if (clusteringBool):
    print (f"Clustering Bool:{clusteringBool}")
    print (f"KMeans Clustering:{clustering}")
else:
    print(f"Clustering Bool:{clusteringBool}")
    print (f"Clustering:{clustering}")
print (f"WSClean_grid:{WSClean_grid}")
print (f"ms_fieldcenter:{ms_fieldcenter}")
print (f"WSClean Path: {wsclean_path}")
print (f"Partitions: {partitions}")



################################################################################################################################################################################
# Control Variables ########################################################################################
###########################################################################################################

# Column Name: Column in MS file to be imaged (DATA is usually uncalibrated, CORRECTED_DATA is calibration and MODEL_DATA contains WSClean model output)
column_name = "DATA"

# IF USING WSCLEAN IMAGE GRID: sampling wrt WSClean grid
# 1 means the output will have same number of pixels as WSClean image
# N means the output will have WSClean Image/N pixels
sampling = 1

# error tolerance for FFT
eps = 1e-3

#precision of calculation
precision = 'single'



#user_fieldcenter: Invoked if WSClean_grid and ms_fieldcenter are False - gives allows custom field center for imaging of specific region
user_fieldcenter = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")

#Time
time_start = 0
time_end = -1
time_slice = 1

# channel
channel_id = np.array([0], dtype = np.int64)
#channel_id = np.array([4,5], dtype = np.int64) # vary the number of partitions - 1 channel works sometimes with 1, sometimes doesn't, 64 channels will need more partitions. 
#channel_id = np.arange(64, dtype = np.int32)

# Create context with selected processing unit.
# Options are "AUTO", "CPU" and "GPU".
ctx = bipp.Context("CPU")

filter_tuple = ('lsq', 'std') # might need to make this a list

std_img_flag = True # put to true if std is passed as a filter

plotList= np.array([1,2])
# 1 is lsq
# 2 is levels
# 3 is WSClean
# 4 is WSClean v/s lsq comparison

# 1 2 and 3 are on the same figure - we can remove 2 and 3 from this figure also
# 4 is on a different figure with 1 and 2

#######################################################################################################################################################
# Observation set up ########################################################################################
#######################################################################################################################################################

#ms = measurement_set.MwaMeasurementSet(ms_file)
#N_antenna = 128 # change this to get this from measurement set
#N_station = 128

try:
    if (channel_id.shape[0] > 1):
        frequency = ms.channels["FREQUENCY"][0] + (ms.channels["FREQUENCY"][-1] - ms.channels["FREQUENCY"][0])/2
        print ("Multi-channel mode with ", channel_id.shape[0], "channels.")
    else: 
        frequency = ms.channels["FREQUENCY"][channel_id]
        print ("Single channel mode.")
except:
    frequency = ms.channels["FREQUENCY"][channel_id]
    print ("Single channel mode.")

wl = constants.speed_of_light / frequency.to_value(u.Hz)
print (f"wl:{wl}; f: {frequency}")

if (WSClean_grid): 
    with fits.open(wsclean_path, mode="readonly", memmap=True, lazy_load_hdus=True) as hdulist:
        cl_WCS = awcs.WCS(hdulist[0].header)
        cl_WCS = cl_WCS.sub(['celestial'])
        cl_WCS = cl_WCS.slice((slice(None, None, sampling), slice(None, None, sampling)))

    field_center = ms.field_center

    width_px, height_px= 2*cl_WCS.wcs.crpix 
    cdelt_x, cdelt_y = cl_WCS.wcs.cdelt 
    FoV = np.deg2rad(abs(cdelt_x*width_px) )
    print ("WSClean Grid used.")
else: 
    if (ms_fieldcenter):
        field_center = ms.field_center
        print ("Self generated grid used based on ms fieldcenter")
    else:
        field_center = user_fieldcenter
        print ("Self generated grid used based on user fieldcenter")

lmn_grid, xyz_grid = frame.make_grids(N_pix, FoV, field_center)

gram = bb_gr.GramBlock(ctx)

print (f"Initial set up takes {tt.time() - start_time} s")

# Nufft Synthesis options

opt = bipp.NufftSynthesisOptions()
# Set the tolerance for NUFFT, which is the maximum relative error.
opt.set_tolerance(eps)
# Set the maximum number of data packages that are processed together after collection.
# A larger number increases memory usage, but usually improves performance.
# If set to "None", an internal heuristic will be used.
opt.set_collect_group_size(None)
# Set the domain splitting methods for image and uvw coordinates.
# Splitting decreases memory usage, but may lead to lower performance.
# Best used with a wide spread of image or uvw coordinates.
# Possible options are "grid", "none" or "auto"
#opt.set_local_image_partition(bipp.Partition.grid([1,1,1]))
#opt.set_local_image_partition(bipp.Partition.none()) # Commented out
#opt.set_local_uvw_partition(bipp.Partition.none()) # Commented out
opt.set_local_image_partition(bipp.Partition.grid([partitions,partitions,1]))
opt.set_local_uvw_partition(bipp.Partition.grid([partitions,partitions,1]))
t1 = tt.time()
#time_slice = 25 ### why is this 25 - ask simon @@@

print("N_pix = ", N_pix)
print("precision = ", precision)
print("Proc = ", ctx.processing_unit)

print (f"Initial set up takes {tt.time() - start_time} s")


########################################################################################
### Intensity Field ########################################################################################
########################################################################################
# Parameter Estimation
########################################################################################
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@ PARAMETER ESTIMATION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
if (clusteringBool):
    if (clustering):
        num_time_steps = 0
        I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=1, ctx=ctx)
        #for t, f, S, uvw_t in ProgressBar(
        for t, f, S in ProgressBar(
                #ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name, return_UVW = True)
                ms.visibilities(channel_id=channel_id, time_id=slice(time_start, time_end, time_slice), column=column_name)
        ):
            wl = constants.speed_of_light / f.to_value(u.Hz)
            XYZ = ms.instrument(t)

            W = ms.beamformer(XYZ, wl)
            G = gram(XYZ, W, wl)
            S, _ = measurement_set.filter_data(S, W)
            I_est.collect(S, G)
            num_time_steps +=1

        print (f"Number of time steps: {num_time_steps}")
        N_eig, intensity_intervals = I_est.infer_parameters()
    else:
        # Set number of eigenvalues to number of eigenimages 
        # and equally divide the data between them 
        N_eig, intensity_intervals = N_level, np.arange(N_level)
else:
    N_eig, intensity_intervals=39, clustering # N_eig still to be obtained from parameter estimator????? IMP # 26 for 083 084 39 for????

print (f"Number of Eigenvalues:{N_eig}, Intensity intervals: {intensity_intervals}")

########################################################################################
# Imaging ########################################################################################
########################################################################################
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ IMAGING @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
imager = bipp.NufftSynthesis(
    ctx,
    opt,
    N_antenna,
    N_station,
    intensity_intervals.shape[0],
    filter_tuple,
    lmn_grid[0],
    lmn_grid[1],
    lmn_grid[2],
    precision
)

for t, f, S in ProgressBar(
        ms.visibilities(channel_id=channel_id, time_id=slice(time_start, time_end, time_slice), column=column_name)
):
    
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    
    S, W = measurement_set.filter_data(S, W)
    
    #UVW_baselines_t = uvw_t
    #uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
    UVW_baselines_t = ms.instrument.baselines(t, uvw=True, field_center=ms.field_center)
    uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)

    print (S.data)

    imager.collect(N_eig, wl, intensity_intervals, W.data, XYZ.data, uvw, S.data)

lsq_image = imager.get("LSQ").reshape((-1, N_pix, N_pix))
if (filter_negative_eigenvalues):
    I_lsq_eq = s2image.Image(lsq_image.reshape(int(N_level),lsq_image.shape[-2], lsq_image.shape[-1]), xyz_grid)
else:
    I_lsq_eq = s2image.Image(lsq_image.reshape(int(N_level) + 1, lsq_image.shape[-2], lsq_image.shape[-1]), xyz_grid)
print("lsq_image.shape =", lsq_image.shape)

if (std_img_flag):
    std_image = imager.get("STD").reshape((-1, N_pix, N_pix))
    if (filter_negative_eigenvalues):
        I_std_eq = s2image.Image(std_image.reshape(int(N_level), std_image.shape[-2], lsq_image.shape[-1]), xyz_grid)
    else:
        I_std_eq = s2image.Image(std_image.reshape(int(N_level) + 1, std_image.shape[-2], std_image.shape[-1]), xyz_grid)
    print("std_image.shape =", std_image.shape)


# Without sensitivity imaging output

t2 = tt.time()

#plot output image

lsq_levels = I_lsq_eq.data # Nlevel, Npix, Npix
lsq_image = lsq_levels.sum(axis = 0)
print (f"Lsq Levels shape:{lsq_levels.shape}")

if (std_img_flag):
    std_levels = I_std_eq.data # Nlevel, Npix, Npix
    std_image = std_levels.sum(axis = 0)

    print (f"STD Levels shape:{std_levels.shape}")
if (3 in plotList or 4 in plotList):
    WSClean_image = fits.getdata(wsclean_path)
    WSClean_image = np.flipud(WSClean_image.reshape(WSClean_image.shape[-2:]))

if (filter_negative_eigenvalues):
    eigenlevels = N_level
else: 
    eigenlevels = N_level + 1

if (4 in plotList):
    if (filter_negative_eigenvalues):
        fig_comp, ax_comp = plt.subplots(int(len(filter_tuple)), 3, figsize = (40,20))
if (1 in plotList):
    fig_out, ax_out = plt.subplots(len(filter_tuple), 1, figsize = (40,20) )
    ax_outList = ax_out.ravel()
if (2 in plotList):
    fig_out, ax_out = plt.subplots(len(filter_tuple), 1 + eigenlevels, figsize = (40,20))
if (3 in plotList):
    fig_out, ax_out = plt.subplots(len(filter_tuple), 2 + eigenlevels, figsize = (40,20))

if ((1 in plotList) or (2 in plotList) or (3 in plotList)):

    # Output LSQ Image

    BBScale = ax_out[0, 0].imshow(lsq_image, cmap = "cubehelix")
    ax_out[0, 0].set_title(r"LSQ IMG")
    divider = make_axes_locatable(ax_out[0, 0])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(BBScale, cax)

    # Output STD Image
    if (std_img_flag):
        BBScale = ax_out[1, 0].imshow(std_image, cmap = "cubehelix")
        ax_out[1, 0].set_title(r"STD IMG")
        divider = make_axes_locatable(ax_out[1, 0])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(BBScale, cax)

    # output eigen levels
    if ((2 in plotList) or (3 in plotList)):
        for i in np.arange(eigenlevels):
            print (f"Loop {i}: vmin:{lsq_levels[i, :, :].max() * std_levels[i, :, :].min()/std_levels[i, :, :].max() if std_img_flag else lsq_levels[i, :, :].min()}")
            print (f"lsq_levels {i}.max():{lsq_levels[i, :, :].max()} std_levels {i}.min():{std_levels[i, :, :].min()} std_levels {i}.max():{std_levels[i, :, :].max()}")

            lsqScale = ax_out[0, i + 1].imshow(lsq_levels[i, :, :], cmap = "cubehelix", \
                        vmin = (lsq_levels[i, :, :].max() * std_levels[i, :, :].min()/std_levels[i, :, :].max() if std_img_flag else lsq_levels[i, :, :].min()))
            ax_out[0, i + 1].set_title(f"Lsq Lvl {i}")
            divider = make_axes_locatable(ax_out[0, i + 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(lsqScale, cax)

            if (std_img_flag):
                stdScale = ax_out[1, i + 1].imshow(std_levels[i, :, :], cmap = "cubehelix")
                ax_out[1, i + 1].set_title(f"Std Lvl {i}")
                divider = make_axes_locatable(ax_out[1, i + 1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(stdScale, cax)

        # WSC Image
    if ((3 in plotList)):
        WSCleanScale = ax_out[0, -1].imshow(WSClean_image, cmap = "cubehelix")
        ax_out[0, -1].set_title(f"WSC IMG")
        divider = make_axes_locatable(ax_out[0, -1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(WSCleanScale, cax)
    
        if (std_img_flag):
            WSCleanScale = ax_out[1, -1].imshow(WSClean_image, cmap = "cubehelix")
            ax_out[1, -1].set_title(f"WSC IMG")
            divider = make_axes_locatable(ax_out[1, -1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(WSCleanScale, cax)

    if (2 not in plotList):
        fig_out.savefig(f"{outName}")
    else:
        fig_out.savefig(f"{outName}_lvls")
        
    print(f"{outName}.png saved.")

# plot WSC, CASA and BB Comparisons here
if ((4 in plotList)):

    fig_comp, ax_comp = plt.subplots(len(filter_tuple), 3, figsize = (40, 30)) # Right now only WSC and BB included, have to include CASA

    # Comparison LSQ IMAGE 
    BBScale = ax_comp[0, 0].imshow(lsq_image, cmap = "RdBu_r")
    ax_comp[0, 0].set_title(r"LSQ IMG")
    divider = make_axes_locatable(ax_comp[0, 0])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(BBScale, cax)

    # Comparison WSClean image
    WSCleanScale = ax_comp[0, -2].imshow(WSClean_image, cmap='RdBu_r')
    ax_comp[0, -2].set_title(f"WSC IMG")
    divider = make_axes_locatable(ax_comp[0, -2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(WSCleanScale, cax)

    # Comparison STD image
    if (std_img_flag):
        BBScale = ax_comp[1, 0].imshow(std_image, cmap = "RdBu_r")
        ax_comp[1, 0].set_title(r"STD IMG")
        divider = make_axes_locatable(ax_comp[1, 0])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(BBScale, cax)

        WSCleanScale = ax_comp[1, -2].imshow(WSClean_image, cmap='RdBu_r')
        ax_comp[1, -2].set_title(f"WSC IMG")
        divider = make_axes_locatable(ax_comp[1, -2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(WSCleanScale, cax)

    # Comparison LSQ-WSC Difference image
    diff_image = lsq_image - WSClean_image
    diff_norm = TwoSlopeNorm(vmin=diff_image.min(), vcenter=0, vmax=diff_image.max())

    diffScale = ax_comp[0, -1].imshow(diff_image, cmap = 'RdBu_r', norm=diff_norm)
    ax_comp[0, -1].set_title("Diff IMG")
    divider = make_axes_locatable(ax_comp[0, -1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(diffScale, cax)

    # Comparison LSQ/WSC - 1 image
    ratio_image = lsq_image/WSClean_image - 1
    clipValue = 2.5
    ratio_image = np.clip(ratio_image, -clipValue, clipValue)
    ratio_norm = TwoSlopeNorm(vmin=ratio_image.min(), vcenter=1, vmax=ratio_image.max())

    ratioScale = ax_comp[1, -1].imshow(ratio_image, cmap = 'RdBu_r', norm=ratio_norm)
    ax_comp[1, -1].set_title(f"Ratio IMG (clipped $\pm$ {clipValue})")
    divider = make_axes_locatable(ax_comp[1, -1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(ratioScale, cax)

    fig_comp.savefig(f"{outName}_comparison")
    print (f"{outName}_comparison.png saved.")

print(f'Elapsed time: {tt.time() - start_time} seconds.')