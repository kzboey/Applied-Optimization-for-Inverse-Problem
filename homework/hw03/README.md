# Homework Sheet 3


## --------Part 1: Setup ----------------------------------------------------------------------------------------------

### Environement:

The environment is setup on a macOS Catalina v10.15.7 and on recsrv03. In homework sheet 3, the results of homework 1 (More gradient based methods) are executed on the remote server (recsrv03). Homework 2 are executed on my local environment.

### Dataset:

Throughout homework 1, the Helsinki tomography challenge the Helsinki Tomography Challenge 2022 open tomographic dataset (HTC 2022) [2] is used. In particular, sinogram 1b (full) from the challenge dataset is used to perform the experiment to demonstrate the methods and findings. In homework 2 (denoising and deblurring), a very common image processing image - Lena [2] of dimension (256x256) is used. The dataset is in the hw03 directory.

### Result reporoducability:

go the the director of hw03, results of Homework 1 can be reproduced by running the script script_server.py on recsrv03. Some parts (minor) of homework 1 is in Hw03.ipynb which will be specifically mentioned. All paths are already configured in config.json to point to the correct dataset on recsrv03. To run the results locally, change the path htc.data_path.local in config.json to the one desired by the user. For homework 2, run the Jupyter notebook file - HW03_Q2.ipynb.

### Images:

All generated image are stored in the folder "img2" inside "hw02" folder. For image generated in homework 2, simply refer to HW03_Q2 notebook. In the following section, reconstructed image will be referred by the names of the image files.

## -----Part 2: Experimental methods and results -----------------------------------------------------------------

The following sections will refer to the detailed method employed for Homework sheet 3 and the resultings results they produce.

###  Homework 1:

All descent methods OGM1, FGM1, vanilla gradient descent (gd), Landweber Itereation, SIRT and conjugate gradient method are implemented in the file GradientDescent.py. Refer to the file for implementation details. Much modification has been made to improve the descent algorithm from the last homework, eg: (i) parameters can now to be pass to one specific descent algorithm, i.e OGM1 to determine the type of regularizers used. (ii) finite difference operator - C is implemented using scipy sparse package to remove error of failing to allocate memory to large array.

First of all, the algorithms are first verified to be a correct. Hence, a small matrix A=[2 1; 3 3] and observation b=[3;5] is used to cross check the the algorithms against the computed solution in python, which is [ 4.65536559 -4.32038662]. The results can be seen in Hw03.ipynb section (iii) Verification of algorithms. All algorithm does converges to the solution albeit with vastly different iteration number. Besides, the calculation of the largest singular value using my implemented power iteration method is also verified against the singular value computed by python, 4.754135216376083 against 4.75413522. 

The tomography problem is formulated into least square problem, 0.5*|Ax-b|^2 to be solved using the aforementiond descent method. A is obtained using the aomip XrayOperator.x is an initalized value, usually with all elements of vector set to -50, this is chosen based on the result I have tested so far, and b is obtained directly from the htc dataset [1]. 

I set out to present only the results from the best parameterers that I manage to achieve thus far.

*note that beta is the general regularization parameter and delta is the parameter for only huber and fair potential function

#### (i) OGM1 vs FGM1 vs vanilla gradient descent

1. OGM1 - lipshitz constant: 50000, initial starting point: all elements set to -50, regularizer: none, iteration: 25, results: OGM_none.png

2. OGM1 - lipshitz constant: 50000, initial starting point: all elements set to -50, regularizer: tikhonov, beta:5, iteration: 25, results: OGM_tikhonov.png

3. OGM1 - lipshitz constant: 50000, initial starting point: all elements set to -50, regularizer: huber, beta:5, delta: 5, iteration: 25, results: OGM_huber.png

4. OGM1 - lipshitz constant: 50000, initial starting point: all elements set to -50, regularizer: fair potential, beta:5, delta: 5, iteration: 25, results: OGM_fair.png

5. gd - learning rate: 0.000005, initial starting point: all elements set to -50, regularizer: none, iteration: 1000, results: gd_vanilla.png

6. gd - learning rate: 0.000005, initial starting point: all elements set to -50, regularizer: tikhonov, beta:5, iteration: 1000, results: gd_tikhonov.png

7. gd - learning rate: 0.000005, initial starting point: all elements set to -50, regularizer: huber, beta:2, delta: 2, iteration: 1000, results: gd_huber.png

8. gd - learning rate: 0.000005, initial starting point: all elements set to -50, regularizer: fair potential, beta:2, delta: 2, iteration: 1000, results: gd_fair.png

9. FGM1 - lipshitz constant: 50000, initial starting point: all elements set to -50, regularizer: none, iteration: 25, results: FGM_none.png

10. FGM1 - lipshitz constant: 50000, initial starting point: all elements set to -50, regularizer: tikhonov, beta:5, iteration: 25, results: FGM_tikhonov.png

11. FGM1 - lipshitz constant: 50000, initial starting point: all elements set to -50, regularizer: huber, beta:5, delta: 5, iteration: 25, results: FGM_huber.png

12. FGM1 - lipshitz constant: 50000, initial starting point: all elements set to -50, regularizer: fair potential, beta:5, delta: 5, iteration: 25, results: FGM_fair.png

As can be seen, OGM1 and FGM1 both gives similar reconstruction with lower iteration than gd without momentum. However, the spots in the middle of the objects are still struggle to reconstructed. the result fomr 8 produce the best score on the leaderboard of approximately 82%.

#### (ii) Landweber iteration

Using the power iteration, the largest singular value of A is calculated to be 20.962801. Thus, the upper bound of the step length should be 2/(20.962801^2) which is approximately 4.55^-3. In my implementation of the landweber iteration algorithm, if the chosen step length exceed this bound, the lambda parameter will be set to the upper bound of the fraction value with only the leading non zero value without rounding, eg, 4.55^-3 will be 4.0^-3.

1. lambda: 0.0001, initial starting point: all elements set to -50, iteration: 1000, result: Landweber.png

Results are not too ideal compared to the methods in (i), but still the overell shape of the object is reconstructed.

#### (iii) SIRT

Implementation detail is similar to above Landweber iteration, just that its multiplied by the additional D and M matrix.

Unfortunately, i get an error when doing the reconstruction but not in the small example matrix.

ValueError: expected 1-d or 2-d array or matrix, got array(<100800x100800 sparse matrix of type '<class 'numpy.float32'>'
	with 100800 stored elements (1 diagonals) in DIAgonal format>,
      dtype=object)
      
Hence, in the future troubleshooting needs to be done to determine the source of this error and to find a solution to make it work.

#### (iv) Conjugate Gradient


initial starting point: all elements set to -50, iteration: 1000, result: cg.png 

Result is rather similar to (ii)


###  Homework 2:

*refer HW03_Q2.ipynb for all image results

#### (i) Denoising

Gaussian noise - 0 mean, standard deviation of 50, added norm error: 31290.1869176628
Poisson noise - lambda 1000, added norm error: 256102.13178925318
salt and pepper noise = randomly set any (256x256) element to the min value, 3 and max value (238) respectively. added norm error: 43282.09062649354

Illustration of image before and noise are added can be seen in the notebook.

The denoisining process is formulated into a least square problem, 0.5*|Ax-b|^2. A is a identity matrix of size 65536, x and b are both vector of size 65536, b is the flatten image with added gaussian noise. 7 experiments are perform and the results are recorded in the notebook. the denoised result are calculated based on the norm difference between the denoised image and original image, and the result is compared against the added norm error by the noise.

1. gd - learning rate: 0.01, initial starting point: all elements set to median of max and min value, regularizer: tikhonov, beta=2, iteration: 670, error: 19557.593880908615

2. OGM1 - Lipschitz constant: 10000, initial starting point: all elements set to median of max and min value, regularizer: tikhonov, beta=2, iteration: 2500, error: 19589.540691015398

3. FGM1 - Lipschitz constant: 10000, initial starting point: all elements set to median of max and min value, regularizer: tikhonov, beta=2, delta=2, iteration: 2500, error: 19627.535269521402

4. OGM1 - Lipschitz constant: 10000, initial starting point: all elements set to median of max and min value, regularizer: huber, beta=2, delta=2, iteration: 2500, error: 12744.781824466094

5. Landweber - lambda: 0.01, initial starting point: all elements set to median of max and min value, iteration: 500, error: 12759.064633303733

6. SIRT - lambda: 0.01, initial starting point: all elements set to median of max and min value, iteration: 500, error: 12759.064633303733

7. CG - initial starting point: all elements set to median of max and min value, iteration: 1, error: 12842.60232494789

From the result, OGM1 with huber functional performs the best. Landweber and SIRT  and CG also performs comparatively well. Although cg can only be run for 1 iteration before the values become explosively large.

#### (ii) Deblurring

See the notebook section 2 for implementation details. The findings is that after adding noise to blurred image and then deblurred it, the noise will also be removed. Visual results of this process are in the notebook.

Next, gd and OGM1 are used to deblur the image, with just 10 iterations, the deblurring effect is already satisfactory.


## -----Part 3: Challenges, limitations, lessons and future improvement ---------------------------------------

The main challenge is the difficulty in parameterizing the descent algorithm, i.e: stepsize, iteration, regularization parameter, hence, it is a big challenge to find the best lipshitz constant, especially for the OGM1 and FGM1 method. In addition finding a suitable number of iteration is also a big challenge.

The main lesson is to find a good methods to determine the parameter, in the exercise random guessing is used which doesn't yield a very good outcome. Perhaps better methods need to be devise to find a set of better parameters.


## -----Part 4: Reference ---------------------------------------------------------------------------------------------

1. Meaney, Alexander, Silva de Moura, Fernando, & Siltanen, Samuli. (2022). Helsinki Tomography Challenge 2022 open tomographic dataset (HTC 2022) (1.2.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7418878
2. http://www.eecs.northwestern.edu/~faisal/d20/d20.html
