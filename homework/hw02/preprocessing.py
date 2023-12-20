import numpy as np

# Part i)
def flatfield_correction(data_projs,data_dark,data_flat):
    ## braodcasting
    data_projs -= data_dark
    data_projs /= (data_flat - data_dark)
    return data_projs

# Part ii)
def get_I0(num_projections, preprocessed_projs,slice_margin=10):
    # function to estimate initial intensity I0
    I0 = np.zeros(num_projections)
    for proj in range(num_projections):
        Im = preprocessed_projs[proj]
        I0[proj] = np.mean([np.mean(Im[:slice_margin, :]),
             np.mean(Im[-slice_margin:, :])])
    return I0

def get_absorption_image(num_projections, projs_rows, projs_cols,preprocessed_projs,I0):
    absorption_image=np.zeros((num_projections, projs_rows, projs_cols), dtype=np.float32)
    for i in range(num_projections):
        absorption_image[i] = transmission_to_absorption(preprocessed_projs[i],I0[i])
    return absorption_image

def transmission_to_absorption(x, I0):
    return -np.log(x / I0)

def absorption_to_transmission(x, I0):
    return I0*np.exp(-x)

# Part iii)
def signal_cleaning(num_projections,projs_rows,projs_cols,absorption_image):
    cleaned_signal = np.zeros((num_projections, projs_rows, projs_cols), dtype=np.float32)
    for i in range(num_projections):
        ai = absorption_image[i]
        
        #truncated absorption signal of negative values
        ai[ai<0]=0
        cleaned_signal[i]=ai
    return cleaned_signal

# Part iv)
def is_power_of_two(n):
    #only allow binning factor of power 2
    if n <= 0:
        return False
    return n & (n - 1) == 0    

## For future, implement kwargs**
def containerised_projections(num_projections, projs_rows, projs_cols, projections, idx, bin_factor):
    # Container for projections
    containerized_projections=np.zeros((num_projections, projs_rows, projs_cols), dtype=np.float32)
    
    for i in range(num_projections):
        if idx == 4:
            containerized_projections[i] = bin_array(projections[i],bin_factor)
    return containerized_projections

def bin_array(X, bin_factor):
    # X can be 1D or 2D array
    if is_power_of_two(bin_factor) == False:
        raise ValueError("binning factor must be power of 2")
    
    elif X.ndim == 1:
        if len(X) % bin_factor != 0:
            raise ValueError(f"Array length ({len(X)}) must be divisible by bin_factor ({bin_factor})")

        n_bins = len(X) // bin_factor
        binned_arr = np.zeros(n_bins)

        for i in range(n_bins):
            start = i * bin_factor
            end = start + bin_factor
            binned_arr[i] = np.mean(X[start:end])

        return binned_arr
    
    elif X.ndim == 2:
        # Determine the size of the blocks
        block_size = bin_factor

        # Calculate the new shape of the array after binning
        new_shape = (X.shape[0] // block_size, X.shape[1] // block_size)

        # Reshape the array into a 2D array of non-overlapping blocks
        blocks = X[:new_shape[0] * block_size, :new_shape[1] * block_size].reshape(
            new_shape[0], block_size, new_shape[1], block_size
        )

        # Take the mean of each block to produce the binned array
        binned_arr = blocks.mean(axis=(1, 3))

        return binned_arr
        

# Part v)

# temporarily unused
# def correct_center_of_rotation_2(projections, roll_pixels):
#     # Roll the projections along the last axis
#     rolled_projections = np.roll(projections, roll_pixels, axis=-1)

#     return rolled_projections

def correct_center_of_rotation(num_projections, projs_rows, projs_cols, projections, roll_pixels): 
    # Container for all rolled projections
    rolled_projections=np.zeros((num_projections, projs_rows, projs_cols), dtype=np.float32)
    
    for i in range(num_projections):
        # Roll the projections along the last axis
        rolled_projections[i] = np.roll(projections[i], roll_pixels, axis=-1)

    return rolled_projections
    
# Part vi)    
def apply_padding(num_projections, projs_rows, projs_cols, projections, pad_size):
    # Get the dimensions of the original image
    height, width = projections[0].shape
    
    # Container for all padded projections
    padded_projections=np.zeros((num_projections, projs_rows + 2 * pad_size, projs_cols + 2 * pad_size), dtype=np.float32)
   
    for i in range(num_projections):
        # Create a new array with the desired padding size
        padded_proj = np.zeros((height + 2 * pad_size, width + 2 * pad_size))

        # Copy the original image into the padded array
        padded_proj[pad_size:-pad_size, pad_size:-pad_size] = projections[i]
        
        padded_projections[i] = padded_proj
    return padded_projections