import casatasks as ct


ct.importfits('/scratch/krishna/GeneratedMSFiles/MWA_CenA_lvl0.fits', imagename='mybippimagelvl0.residual',overwrite=True)
ct.importfits('/scratch/krishna/GeneratedMSFiles/MWA_CenA-psf.fits', imagename='mybippimagelvl0.psf',overwrite=True)
ct.deconvolve(imagename='mybippimagelvl0', deconvolver='multiscale', niter=1000000, mask="MWA_CenA_lvl0.mask", scales=[0,12,24,49,98,196,392],
           gain=0.8, threshold='0.000001Jy', verbose=True)
ct.exportfits(imagename='mybippimagelvl0.image',fitsimage='MWA_CenA_BIPP_lvl0-image.fits',overwrite=True)
ct.exportfits(imagename='mybippimagelvl0.model',fitsimage='MWA_CenA_BIPP_lvl0-model.fits',overwrite=True)
ct.exportfits(imagename='mybippimagelvl0.residual',fitsimage='MWA_CenA_BIPP_lvl0-residual.fits',overwrite=True)