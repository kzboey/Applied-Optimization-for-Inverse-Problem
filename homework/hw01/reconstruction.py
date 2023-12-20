import numpy as np
from functools import partial
from scipy.interpolate import interp1d

"""
    Steps:
    0. Given a sinogram, by first applying radon transform and some filter in the frequency domain
    1. (Might) Add padding to the sinogram to apply the FFT, make it even or power of 2
    2. For each projection, transform it into the frequency domain using the Fast Fourier Transform,
       multiply it coefficient wise with the filter, 
    3. transform the result back using the Inverse Fast Fourier Transform.
    4. Apply Ram-Lak filter, a high pass filter, with a linear response, and implement also two other filters.
    5. First, you should use NumPy’s implementation of the FFT.    
"""

def _get_fourier_filter(size, filter_name):
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # Computing the ramp filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes
    # small bias 
    fourier_filter = 2 * np.real(np.fft.fft(f))         # ramp filter
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = np.pi * np.fft.fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = np.fft.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter

    return fourier_filter[:, np.newaxis]

def _get_fbp(radon_image, theta, filter_name="ramp"):
    img_shape = radon_image.shape[0]
    output_size = img_shape
    angles_count = len(theta)
    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Apply filter in Fourier domain
    if filter_name=="none":
        projection = np.fft.fft(img, axis=0) 
    else:    
        fourier_filter = _get_fourier_filter(projection_size_padded, "ramp")
        projection = np.fft.fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(np.fft.ifft(projection, axis=0)[:img_shape, :])

    # Reconstruct image by interpolation
    reconstructed = np.zeros((output_size, output_size),
                             dtype=radon_image.dtype)
    radius = output_size // 2
    xpr, ypr = np.mgrid[:output_size, :output_size] - radius
    x = np.arange(img_shape) - img_shape // 2
    for col, angle in zip(radon_filtered.T, np.deg2rad(theta)):
        t = ypr * np.cos(angle) - xpr * np.sin(angle)
        interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)  
        reconstructed += interpolant(t)

    return reconstructed * np.pi / (2 * angles_count)

def mse(image1, image2):
    error = np.mean((image1 - image2)**2)
    return error