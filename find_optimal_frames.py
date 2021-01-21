# ---------------------------------------------------------------------
#  File: find_optimal_frames.py
#  Author: Jan Kukacka
#  Date: 1/2021
# ---------------------------------------------------------------------
#  Package interface function for finding optimal frames
# ---------------------------------------------------------------------

import os
import argparse
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import metadata
import motion_metrics
import ultrasound_utils
from pathlib import Path


# argument parsing
parser = argparse.ArgumentParser(description='FIND OPTIMAL FRAMES: Tool for motion analysis of MSOT Scans')
# required_named = parser.add_argument_group('Required arguments')
# required_named.add_argument('--dataset', type=str, help='generate results for which dataset', required=True)
parser.add_argument('input_filename', type=str, help='Filename of the file with list of scans to process')

parser.add_argument('-v', '--verbose', help='Print results to the output', action='store_true', default=argparse.SUPPRESS)
parser.add_argument('-g', '--output_gifs', help='Save animation of results as gif', action='store_true', default=argparse.SUPPRESS)
parser.add_argument('-t', '--output_txt', help='Save results as text file', action='store_true', default=argparse.SUPPRESS)
parser.add_argument('-p', '--output_plot', help='Save motion plot as image', action='store_true', default=argparse.SUPPRESS)
parser.add_argument('--strict', help='Strict mode - only consider whole frames', action='store_true', default=argparse.SUPPRESS)
parser.add_argument('--agg', type=str, help='Type of aggregation of scores. "mean", "max", or "75percentile". Default is "mean".', default=argparse.SUPPRESS)
parser.add_argument('--n_best', type=int, help='Maximal number of positions to consider. Default = 5', default=argparse.SUPPRESS)
parser.add_argument('--min_len', type=int, help='Minimal number of ultrasound frames to consider an msot frame. Default = 4', default=argparse.SUPPRESS)
parser.add_argument('--distance', type=int, help='Minimal number of positions between neighboring peaks.', default=argparse.SUPPRESS)
parser.add_argument('--metrics', type=str, help='List of metrics to use.', nargs='*', default=argparse.SUPPRESS)


def find_optimal_frames(scans, verbose=True, strict=None, metrics=None, agg=None,
                        distance=None, min_len=None, n_best=5, output_txt=False,
                        output_gifs=False, output_plot=False):
    '''
    This function takes a list of scans to process and for each of them finds
    an optimal position(s) based on motion in the ultrasound sequences.

    # Arguments:
        - scans: list of scans to process. A scan is represented by a path to
            the folder where it resides, which is expected to contain an .msot
            and an .us file. Alternatively, a list of tuples can be passed, where
            first tuple element is a path to the .msot file and the second is
            a path to the .us file corresponding to one scan.

    '''
    results = {}
    for scan in scans:
        ## Decide if scan is a directory path or tuple of two filenames
        try:
            files = os.listdir(scan)
            scan = Path(scan)
            filename_msot = scan / [file for file in files if file.lower().endswith('.msot')][0]
            filename_us = scan / [file for file in files if file.lower().endswith('.us')][0]
            output_folder = scan / ('motion_analysis_' + filename_us.stem)
        except IndexError:
            ## Looks like files were not found
            raise('In folder ' + str(scan) + '.msot and .us file could not be found.')
        except TypeError:
            ## Looks like scan is a tuple
            filename_msot, filename_us = scan
            output_folder = Path(filename_msot).parent / ('motion_analysis_' + Path(filename_msot).stem)

        if output_gifs or output_txt or output_plot:
            ensure_folder(output_folder)

        ## Load ultrasound once - multiple functions use it
        us = ultrasound_utils.load_acuity_ultrasound(filename_us,
                                                     replace_zeros=True)

        ## Compute optimal positions
        optimal_positions, motion_scores = process_scan(us, filename_msot, strict=strict,
                                                        distance=distance, metrics=metrics,
                                                        agg=agg, min_len=min_len)

        n_wavelengths = metadata.get_wavelength_count(filename_msot)

        ## Optionally, print results
        if verbose:
            print_results(motion_scores, optimal_positions, filename_msot, strict)

        optimal_positions = optimal_positions[:n_best]

        ## Optionally, save gifs
        if output_gifs:
            save_gif(us, filename_msot, optimal_positions, output_folder, strict)

        ## Optionally, save results to text file
        if output_txt:
            kwargs = {'strict': strict, 'metrics': metrics, 'agg': agg,
                      'distance': distance, 'min_len': min_len, 'n_best': n_best}
            save_txt(motion_scores, optimal_positions, output_folder, kwargs)

        ## Optionally, save motion plots
        if output_plot:
            save_plot(motion_scores, optimal_positions, output_folder)

        results[scan] = optimal_positions
    return results


def quantify_motion(us, metrics=None, max_range=1):
    '''
    Evaluates (specified) motion metrics on a given ultrasound image sequence.

    # Arguments:
    - `us`: numpy array of shape `(n_frames, height, width)` with the ultrasound
        image sequence.
    - `metrics`: list of metric codes. See `motion_metrics._all_metrics`
        for available codes. Uses `['znxc', 'ssim']` by default.
    - `max_range`: positive int. Defines maximum range of ultrasound frames
        between which motion will be quantified.

    # Returns:
    - `results`: dict of format `{metric_name: metric_results, ...}`. Metric
        names are same as keys given in argument `metrics`. Metric results are
        numpy arrays of shape `(max_range, n_frames-1)` containing results
        of evaluating given metric between two frames, i.e.
            `metric_results[i, j] = metric(us[j], us[j+i+1])`
        Invalid elements (i.e. where `j+i+1 > n_frames`) are set to `np.nan`.
    '''
    if metrics is None:
        metrics = ['znxc', 'ssim']

    results = {}
    for metric in metrics:
        ## Init empty array with nan-s to mark empty elements
        result = np.full((max_range, len(us)-1), np.nan)
        for i in range(max_range):
            for j in range(len(us)-1-i):
                result[i,j] = motion_metrics._all_metrics[metric](us[j], us[j+1+i])

        results[metric] = result
    return results


def metrics_to_ranks(metrics, normalize=True):
    '''
    Converts computed metric scores to rank scores.

    # Arguments
    - `metrics`: Dictionary with results of motion quantification via various
        metrics. See `quantify_motion` for details.
    - `normalize`: `bool`. If True, rank scores are divided by sequence length
        to range from 0 to 1. Useful to make motion scores computed on various
        lentht sequences comparable.

    # Returns
    - `ranks`: Dictionary with the same format as `metrics` containing rank scores
    '''
    ## Double argsort returns ranks
    ranks = {metric: score.argsort().argsort() for metric,score in metrics.items()}
    if normalize:
        ranks = {metric: score / score.shape[1] for metric,score in ranks.items()}
    return ranks


def aggregate_scores(metrics, frame_ranges, agg=None, min_len=None):
    '''
    Aggregate scores for individual MSOT frames.

    # Arguments
    - `metrics`: Dictionary with computed metrics quantifying motion between
        ultrasound frames. For details see `quantify_motion`.
    - `frame_ranges`: List of ultrasound ranges corresponding to each evaluated
        MSOT frame. Ranges are represented as a tuple `(start_frame, end_frame)`
        containing indices to the ultrasound image sequence. Frame indices are
        inclusive, i.e. the range would be selected as:
            `us[start_frame:stop_frame+1]`
    - `agg`: aggregation method. Defines how are motion scores within one MSOT
        frame aggregated to a single number. Supported values are `'mean'`,
        `'max'`, or `'75percentile'`.
    - `min_len`: positive int. Minimum length of range to consider. Frames at
        the beginning may have a single US image and thus not produce a robust
        estimate. Default is 4.

    # Returns
    - `results`: Numpy array of length `len(frame_ranges)` with aggregated
        scores for each frame range.
    '''
    if agg is None:
        agg = 'mean'
    if min_len is None:
        min_len = 4

    results = np.zeros(len(frame_ranges))
    for i, (start, stop) in enumerate(frame_ranges):
        range_len = stop - start + 1
        if start < 0 or range_len < min_len:
            results[i] = np.inf
            continue

        for metric, score in metrics.items():
            ## Select a triangle of relevant scores
            tri_score = score[range_len::-1, start:stop+1][np.where(np.tri(range_len))]

            if agg == 'mean':
                results[i] += np.mean(tri_score)
            elif agg == '75percentile':
                results[i] += np.percentile(tri_score,75)
            elif agg == 'max':
                results[i] += np.max(tri_score)
            else:
                raise ValueError('Invalid argument "agg". Must be "mean", "max"'
                                 ', or "75percentile".')
        results[i] /= len(metrics)
    return results


def print_results(results, peaks, metadata_filename, strict=None):
    '''
    Prints results of the optimal position selection algorithm in a formatted
    table.

    # Arguments
    - `results`: numpy array with results of shape `(n_frames,)`
    - `peaks`: list of best position indices sorted by motion score
    - `metadata_filename`: full path to the .msot metadata file
    - `strict`: `bool`. If `True`, results include only whole MSOT frames.
    '''
    if strict is None:
        strict = False

    ## Compute frame ranges - same as in process_scan
    frame_matches = metadata.get_oa_to_us_match(metadata_filename)
    n_wavelengths = metadata.get_wavelength_count(metadata_filename)
    ranges = np.stack([frame_matches[:-n_wavelengths+1],
                       frame_matches[n_wavelengths-1:]], axis=1)
    if strict:
        ranges = ranges[::n_wavelengths]

    print_oa_range = lambda peak,n_wavelengths: f'[{peak//n_wavelengths:>3} ({peak%n_wavelengths:>2}) - {(peak+n_wavelengths-1)//n_wavelengths:>3} ({(peak+n_wavelengths-1)%n_wavelengths:>2})]  '
    if strict:
        print_oa_range = lambda peak,n_wavelengths: f'[{peak:>3} ( 0) - {peak:>3} ({n_wavelengths:>2})]  '
    print('Peak  Position  US range     OA range               Score ')
    print('-'*58)
    for i,peak in enumerate(peaks):
        print(f'{i:>4}  {peak:>8}  [{ranges[peak][0]:<4}-{ranges[peak][1]:>4}]  '
              + print_oa_range(peak, n_wavelengths)
              + f'{results[peak]:.4f}')
    print('-'*58)
    print(f'Median{np.median(results[np.isfinite(results)]):>52.4f}')
    print(f'Mean{np.mean(results[np.isfinite(results)]):>54.4f}')


def save_plot(results, peaks, output_folder):
    '''
    Function to render the results as a plot with marked optimal positions and
    save it to the desired output folder.

    # Arguments
    - `results`: numpy array with results of shape `(n_frames,)`
    - `peaks`: list of best position indices sorted by motion score
    - `output_folder`: Path where to save the output
    '''
    output_folder = Path(output_folder)

    fig,ax = plt.subplots()
    ax.plot(results)
    ax.plot(peaks,results[peaks],'go')
    fig.savefig(output_folder / 'motion_profile.png', bbox_inches='tight')
    plt.close(fig)


def save_gif(us, metadata_filename, peaks, output_folder, strict=None):
    '''
    Function to render the results as a plot with marked optimal positions and
    save it to the desired output folder.

    # Arguments
    - `us`: numpy array with sequence of ultrasound images, shape
        `(n_frames, height, width)`
    - `metadata_filename`: Filename with path to the .msot file corresponding
        to the analyzed scan
    - `peaks`: list of best position indices sorted by motion score
    - `strict`:
    - `output_folder`: Path where to save the output
    '''
    if strict is None:
        strict = False

    try:
        import happy as hp
    except ImportError:
        print('To save gifs, library "happy" is required.')

    output_folder = Path(output_folder)

    n_wavelengths = metadata.get_wavelength_count(metadata_filename)
    oa_to_us = metadata.get_oa_to_us_match(metadata_filename)

    for i, index in enumerate(peaks):
        us_start = oa_to_us[index]
        us_stop = oa_to_us[index+n_wavelengths-1]
        if strict:
            us_start = oa_to_us[index*n_wavelengths]
            us_stop = oa_to_us[(index+1)*n_wavelengths-1]
        print('gif:', us_start, us_stop)
        animation = us[us_start:us_stop+1]
        animation = ((animation - animation.min()) / animation.ptp() * 255).astype(np.uint8)

        hp.io.save(output_folder / f'pos_{i}_{index}-{index+n_wavelengths-1}.gif',
                   animation, parents=True, overwrite=True)


def save_txt(results, peaks, output_folder, kwargs):
    '''
    Function to save the results to a text file.

    # Arguments
    - `results`: numpy array with results of shape `(n_frames,)`
    - `peaks`: list of best position indices sorted by motion score
    - `output_folder`: Path where to save the output
    - `kwargs`: Set of parameters passed to the parent function.
    '''
    output_folder = Path(output_folder)

    with open(output_folder / "optimal_positions.txt", "w") as output_file:
        print('# Positions', file=output_file)
        print('# position_start; score', file=output_file)
        for peak in peaks:
            print(f'{peak}; {results[peak]}', file=output_file)
        print('# Arguments', file=output_file)
        for name, value in kwargs.items():
            print(f'{name}:', repr(value), file=output_file)


def process_scan(ultrasound, metadata_filename, strict=None,
                 metrics=None, agg=None, distance=None, min_len=None):
    '''
    Takes a scan and sorts frame sequences by their motion. Evaluates the motion
    not only between sequential US frames but between every pair within the
    range corresponding to one multispectral OA frame. This prevents problems
    with slow drifting motion.

    # Arguments
    - `ultrasound`: Either a full path to the ultrasound file OR a
        sequence of ultrasound images.
    - `strict`: `bool`. If `True`, looks only for full multispectral frames. If
        `False` (default), looks for a full set of wavelengths that can
        originate from adjacent multispectral frames.
    - `metrics`: See `quantify_motion`.
    - `agg`: Score aggregation method. See `aggregate_scores`.
    - `distance`: `float`. Higher number means peaks can be closer together.
    - `min_len`: See `aggregate_scores`.
    - `verbose`: `bool`. If `False`, printed results are suppressed
    - `plot`: `bool`. If `False`, plotted results are suppressed

    # Returns
    - `sorted_peaks`: List of indices to optimal positions, sorted from the best
    - `results`: numpy array with motion scores for each position
    '''
    if distance is None:
        distance = 20
    if strict is None:
        strict = False

    ## Try if ultrasound_filename is already loaded ultrasound sequence
    try:
        if ultrasound.ndim == 3:
            us = ultrasound
    except AttributeError:
        ## If not, assume a filename and load it
        us = ultrasound_utils.load_acuity_ultrasound(ultrasound,
                                                     replace_zeros=True)

    frame_matches = metadata.get_oa_to_us_match(metadata_filename)
    n_wavelengths = metadata.get_wavelength_count(metadata_filename)
    ranges = np.stack([frame_matches[:-n_wavelengths+1],
                       frame_matches[n_wavelengths-1:]], axis=1)

    if strict:
        ranges = ranges[::n_wavelengths]

    max_range = np.max(ranges[:,1] - ranges[:,0]) + 1

    scores = quantify_motion(us, metrics, max_range=max_range)
    rank_scores = metrics_to_ranks(scores)

    ## Aggregate metrics
    results = aggregate_scores(rank_scores, ranges, agg, min_len)

    ## Find peaks
    peaks = scipy.signal.find_peaks(-results, distance=len(results)//distance)[0]
    sorted_peaks = peaks[np.argsort(results[peaks])]

    return sorted_peaks, results


def ensure_folder(path):
    '''Make sure a folder exists. Returns False if it cannot be created.'''
    import os
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError as e:
            # print "Couldn't create output directory."
            return False
    return True


def print_intro(kwargs):
    import math
    def center_text(line, width=57):
        len_diff = width - len(line)
        return (' ' * int(math.ceil(len_diff/2)) +
                line +
                ' ' * int(math.floor(len_diff/2)))
    def rpad_text(line, width=57):
        len_diff = width - len(line)
        return line + ' ' * len_diff


    print('╔'+ '═'*57 + '╗')
    print('║'+center_text('FIND OPTIMAL FRAMES')+ '║')
    print('║'+center_text('Tool for motion analysis in MSOT scans')+ '║')
    print('║'+' '*57+ '║')
    print('║'+center_text('Jan Kukacka, 2021') + '║')
    print('╠'+'═'*57 + '╣')

    ## Strict mode
    if 'strict' in kwargs:
        print('║'+rpad_text(' STRICT MODE')+'║')
        print('║'+rpad_text(' Only considers whole MSOT frames. Does not allow')+'║')
        print('║'+rpad_text(' solutions overlapping over two MSOT frames.')+'║')
        print('║'+' '*57+ '║')
    else:
        print('║'+rpad_text(' PROGRESSIVE MODE')+'║')
        print('║'+rpad_text(' Allows solutions overlapping over two MSOT frames.')+'║')
        print('║'+rpad_text(' To evaluate only full frames, run with "--strict".')+'║')
        print('║'+' '*57+ '║')

    ## Computation options
    args = ['metrics', 'distance', 'agg', 'min_len']
    titles = ['Metrics:', 'Peak distance:', 'Score aggregation:',
              'Minimal US seq. length:']

    args_titles = [(x,y) for x,y in zip(args, titles) if x in kwargs]
    n_items = len(args_titles)
    if n_items > 0:
        print('║'+rpad_text(' COMPUTATION OPTIONS')+'║')
        max_len = max([len(t) for _,t in args_titles])
        for i, (arg,title) in enumerate(args_titles):
            c = '├' if i+1<n_items else '└'
            if arg == 'metrics':
                arg_value = ', '.join(kwargs[arg])
            else:
                arg_value = str(kwargs[arg])
            print('║'+rpad_text(f' {c}─ ' + title, 4+max_len) + ' ' + rpad_text(arg_value, 57-5-max_len)+'║')
        print('║'+rpad_text(' ')+ '║')


    ## Reporting options
    args = ['n_best', 'verbose', 'output_txt', 'output_plot', 'output_gifs']
    titles = ['Maximal number of peaks to report: ', '(-v) Print output to console',
              '(-t) Save results to text file', '(-p) Save motion analysis plot',
              '(-g) Save gif with optimal position']

    args_titles = [(x,y) for x,y in zip(args, titles) if x in kwargs]
    n_items = len(args_titles)
    if n_items > 0:
        print('║'+rpad_text(' REPORTING OPTIONS')+'║')
        for i, (arg,title) in enumerate(args_titles):
            c = '├' if i+1<n_items else '└'
            if arg == 'n_best':
                title = title + str(kwargs[arg])
            print('║'+rpad_text(f' {c}─ ' + title, 57)+'║')

    print('╚'+'═'*57 + '╝')
    print()


if __name__ == '__main__':
    import sys
    kwargs = vars(parser.parse_args(sys.argv[1:]))

    print_intro(kwargs)

    if ('output_txt' not in kwargs and 'verbose' not in kwargs and
        'output_gifs' not in kwargs and 'output_plot' not in kwargs):
        print('WARNING: No option for reporting results was specified.')
        print('Run again with some of the flags: -v -t -g -p')
        print('Run with -h to see help.')
        exit()
    with open(kwargs['input_filename'], 'r') as input_file:
        ## Read lines of the input file
        # scans = list(input_file)
        scans = input_file.read().splitlines()

    for i in range(len(scans)):
        if ',' in scans[i]:
            scans[i] = tuple(scans[i].split(','))

    del kwargs['input_filename']

    find_optimal_frames(scans, **kwargs)
