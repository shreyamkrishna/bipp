from casatasks import imstat, immath

# Step 1. Determine the noise level (rms) from your dirty image.
# Optionally, you can specify a region (e.g., a source-free area) with the box or region parameter.
stats = imstat(imagename='MWA_CenA_lvl0.fits')
rms = stats['rms'][0]
print("Measured RMS =", rms)

# Step 2. Define a threshold. For example, set the threshold to 3 times the rms:
threshold_value = 1 * rms
print("Using threshold =", threshold_value)

# Step 3. Create a mask image by thresholding the dirty image.
# The expression 'iif(IM0 > X, 1, 0)' returns 1 for pixels above X and 0 otherwise.
expr = 'iif(IM0 > {thr}, 1, 0)'.format(thr=threshold_value)

immath(imagename='MWA_CenA_lvl0.fits',
       mode='evalexpr',
       expr=expr,
       outfile='MWA_CenA_lvl0.mask')