#!/usr/bin/env python
import sys
from astropy.io import fits

def copy_fits_header(input_fits, output_fits):
    """
    Reads the header from the input FITS file and writes a new FITS file
    with an identical header (and no image data, unless you choose to include it).
    """
    # Open the input FITS file and extract a copy of the primary header
    with fits.open(input_fits) as hdul:
        orig_header = hdul[0].header.copy()  # Copy to avoid modifying the original header
        # Optionally, if you want to also copy the data from the primary HDU:
        # data = hdul[0].data
        # For an empty data array, simply set data = None
        
    # Create a new Primary HDU using the copied header
    new_hdu = fits.PrimaryHDU(header=orig_header)
    # If you want to include data, do:
    # new_hdu = fits.PrimaryHDU(data=data, header=orig_header)
    
    # Write out the new FITS file; overwrite if it already exists
    new_hdu.writeto(output_fits, overwrite=True)
    print(f"Created '{output_fits}' with an identical header from '{input_fits}'.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python copy_header.py <input_fits_file> <output_fits_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    copy_fits_header(input_file, output_file)
