# ------------------------------------------------------------------------------
#  File: ultrasound_utils.py
#  Author: Jan Kukacka
#  Date: 12/2020
# ------------------------------------------------------------------------------
#  Helping functions for loading and processing of the ultrasound images
# ------------------------------------------------------------------------------

import numpy as np


def load_acuity_ultrasound(filename, ultrasound_shape=(200,200),
                           replace_zeros=False):
    '''
    Loads ultrasound image produced by acuity in the '.us' format and returns
    a numpy array.

    Based on old matlab code:
    ```
    function us_stack = loadUS(obj, dirpath,usFileName)
        FID = fopen (fullfile(dirpath,usFileName));
        us=fread (FID, 'uint32' );
        US = reshape(us,obj.us_size,obj.us_size,size(us,1)./(obj.us_size*obj.us_size));
        fclose(FID)  ;
    end
    ```

    # Arguments
    - `filename`: path to the .us file
    - `ultrasound_shape`: size of the ultrasound images, `(height, width)`.
        Default is `(200,200)`
    - `replace_zeros`: `bool`. If `True`, "dead" pixels will be replaced with
        frame mean.

    # Returns
    - `ultrasound`: numpy array with the ultrasound data.
        Shape `(n_frames,height,width)`.
    '''
    with open(filename, 'rb') as file:
        us_image = np.frombuffer(file.read(), dtype=np.uint32)

    us_image = us_image.reshape(-1, *ultrasound_shape)

    ## Acuity images are rotated and upside down
    us_image = us_image.transpose(0,2,1)[:,::-1]

    ## Sometimes US images contain "dead" pixels with zero value which
    ## complicates rendering and processing. Replacing them with frame mean is
    ## a simple fix.
    if replace_zeros:
        us_image = np.where(us_image, us_image,
                            np.mean(us_image, axis=(1,2), keepdims=True))

    return us_image
