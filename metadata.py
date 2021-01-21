# ---------------------------------------------------------------------
#  File: metadata.py
#  Author: Jan Kukacka
#  Date: 1/2021
# ---------------------------------------------------------------------
#  Functions for extracting metadata from scan files
# ---------------------------------------------------------------------

import xml.etree.ElementTree as ET
from functools import lru_cache


@lru_cache(100)
def get_wavelength_count(metadata_filename):
    '''
    Extracts wavelength count for a scan from its metadata.

    # Arguments
    - metadata_fileaname: full filename (including path) to a xml metadata file
        (usually Scan_n.msot).
    '''
    meta = ET.parse(metadata_filename)
    scan_node = meta.getroot().find('ScanNode')
    n_wavelengths = len(list(scan_node.find('Wavelengths').iter('Wavelength')))
    return n_wavelengths


@lru_cache(20)
def get_oa_to_us_match(metadata_filename):
    '''
    Function reads the given metadata file and for each oa pulse returns a
    matching US frame.

    # Arguments
    - metadata_fileaname: full filename (including path) to a xml metadata file
        (usually Scan_n.msot).

    # Returns
    - list of length n_pulses with indices to corresponding US frames.
        Negative numbers mean no matching US frame exists. US frame numbers
        use zero-based indexing.
    '''
    meta = ET.parse(metadata_filename)
    scan_node = meta.getroot().find('ScanNode')
    n_wavelengths = len(list(scan_node.find('Wavelengths').iter('Wavelength')))
    frames = scan_node.find('ScanFrames')
    result = []
    for frame_index, frame in enumerate(frames.iter('DataModelScanFrame')):
        ## Get US frame for each single-wavelenth OA image
        us_frame_index = frame.find('Frame').get('ultraSound-frame-offset')
        us_frame_index = int(us_frame_index) - 1
        result.append(us_frame_index)
    return result
