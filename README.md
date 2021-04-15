# Optimal frames
Library / tool for motion analysis in MS-OPUS scans

# Usage

```
python find_optimal_frames.py input_file [-v -g -t -p -c] [--strict] [--agg {mean|max|75percentile}] [--n_best N] [--min_len N] [--distance N] [--metrics [metric, ...]]

or 

python -m optimal_frames.find_optimal_frames [rest as above]
```

`input_file` contains paths to scans that should be processed. Two formats are possible and can be mixed:

1. 1 folder per line: `D:/Data/MSOT/Study_35/Scan_11`: The folder has to contain a single `.msot` and a single `.us` files.
2. 2 files per line: `D:/Data/MSOT/Study_35/Scan_11/Scan_11.msot,D:/Data/MSOT/Study_35/Scan_11/Scan_11.us`: The first file is the `.msot`, the other is the `.us`.

## Output modes

Flags `-v -g -t -p -c` specify which outputs should the program produce. At least one has to be specified, and multiple can be used at the same time.

> **Pro-tip**: Just use `-vgtpc` to generate all outputs and don't worry about selecting the correct one.

* `-v`: Verbose mode. Results are printed to console directly.
* `-g`: Generate gifs. Produces animated gifs with ultrasound sequences matching selected positions.
* `-t`: Generate a text file. Contains also parameters used for the motion analysis so the results can be reproduced.
* `-p`: Generate motion profile plots.
* `-c`: Generate a csv formatted as input to the MSOT reconstruction code.

Output of flags `g`, `t`, and `p` is saved at the data location in a new folder `motion_analysis_Scan_N`. Output of flag `c` is saved as `motion_analysis.csv` at the directory where the program is executed.

## Other flags

* `--strict`: If specified, the program uses a "strict" mode and only searches for frames aligned with the wavelength sequences. Without the flag, the program evaluates also overlapping frames, i.e. part of the wavelengths comes from one frame and part from a previous one.
* `--n_best N`: Specifies maximum number of optimal positions to find. Defaults to 5.
* `--min_len N`: Speicifies a minimal number of ultrasound frames corresponding to one MSOT frame to consider the MSOT frame. Frames with too few ultrasound frames, occuring at the start of the scanning sequence, can have biased results. Default is 4, but this may fail with scans that only use very few wavelengths.
* `--distance N`: Specifies minimum distance between the optimal positions, to ensure diversity of results. Actual minimal distance is computed as `len(scan)/distance`. Default is 20.
* `--metrics metric metric ...`: Specifies which metrics to use. Possible keys are:
    * 'xc': cross correlation
    * 'nxc': normalized cross correlation
    * 'znxc': zeroed normalized cross correlation
    * 'rmse': root mean squared error
    * 'wass': wasserstein distance (slow!)
    * 'ssim': structural similarity
    * 'nmi': normalized mutual information

    Default is 'znxc' and 'ssim'.
    
# Dependencies

The program requires `numpy`, `scipy`, and `scikit-image`.
