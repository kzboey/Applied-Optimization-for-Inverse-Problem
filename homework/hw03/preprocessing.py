def is_power_of_two(n):
    #only allow binning factor of power 2
    if n <= 0:
        return False
    return n & (n - 1) == 0    

def bin_array(X, bin_factor):
    # X can be 1D or 2D array
    if is_power_of_two(bin_factor) == False:
        raise ValueError("binning factor must be power of 2")
    
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
        
def apply_padding(image, pad_size):
    # Get the dimensions of the original image
    height, width = image.shape
    
    # Container for all padded projections
    padded_image=np.zeros((projs_rows + 2 * pad_size, projs_cols + 2 * pad_size), dtype=np.float32)
   
    # Create a new array with the desired padding size
    padded_image = np.zeros((height + 2 * pad_size, width + 2 * pad_size))

    # Copy the original image into the padded array
    padded_image[pad_size:-pad_size, pad_size:-pad_size] = projections[i]
    
    return padded_image
