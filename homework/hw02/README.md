# Homework Sheet 2


## --------Part 1: Setup ----------------------------------------------------------------------------------------------

### Environement:

The environment is setup on a macOS Catalina v10.15.7 and on recsrv03. In homework sheet 2, the results of homework 1,2,3,5 are executed on my local MacOs environment and only homework 4 is executed on the GPU on the remote server (recsrv03). The reason is because the reconstruction result is differs greatly between a CPU and GPU. Problem 4 is an exception because the magnitude of the calculation can only be executed on a GPU. This problem was discoverd rather late in the week and time constraints this week disallows me to troubleshoot the numerical error resulting from the calculation on GPU which results in bad reconstruction, but this is a good lesson learnt about the difference between performing computation on GPU and CPU.

### Dataset:

Throughout the homework, I am using the Cone-Beam Computed Tomography Dataset of a Seashell [1] for Homework 1,3,4,5. In the Helsinki tomography challenge the Helsinki Tomography Challenge 2022 open tomographic dataset (HTC 2022) [2] is used.

### Result reporoducability:

Results of Homework 1,2,3,5 can be reproduced with given Jupyter notebook HW02_q1345.ipynb in the hw02 folder. Simply change the variable data_path in the 2nd code block and insert the directory of [1] and the notebook should run. Q2 can be reproduced by running HW02_q2.ipynb, but sympy package needs to be installed as a prerequisite. Homework 4 has to be run on a GPU, again change the variable data_path of the script script_v3.py to the directory of the dataset [1] and execute the python file: python script_v3.py. The challenge part can be reproduced by running the notebook HW02_challenge.ipynb. Again change the variable data_path to the directory of dataset[2].

### Images:

All generated image are stored in the folder "Img" inside "hw02" folder. In the following section, reconstructed image will be referred by the names of the image files.

## -----Part 2: Experimental methods and results -----------------------------------------------------------------

The following sections will refer to the detailed method employed for Homework sheet 2 and the findings from the homeworks.

###  Homework 1:

The dataset [1] consists of 721 projection images, and the dimension of each projetion image is (2368x2240). As this is too large to be handled, binning with a factor of 8 is used to shrink the projection image, resulting in a new dimension of (296x280). Then, 3 sinograms are obtained by slicing row 50, 150 and 200 of each projection images and stacked them together, see sliced_sinogram_50.png, sliced_sinogram_100.png and sliced_sinogram_200.png in the image folder. Since there are 721 projection images and the size of each row of binned projection is 280. The dimension of the sinogram is (721x280). 3 different sliced sinogram is made to show the contrast between the upper, middle and lower part of a projcetion image. In subsequent section, sliced sinogram of row 150 will be used throughout to compute the reconstruction because the middle part of a projection image captures the biggest part of the scanned object, in this case, a seashell.

For more details on implementation, see HW02_q1345.ipynb of the section "Homework 1: Preprocessing again".

###  Homework 2:

As its required that it should be able to work for arbitrary 1 or 2 dimension function. The sympy package is used to calculate the gradient of function, as the gradient serves as the direction of update of gradient descent (gd). The 1D case is a simple quadratic function x^2 - x + 1 and 2D case is a convex function x^2 + y^2, so the minimum point can be illustrated more clearly. However, arbitrary scalar function can be used by change the variable f1d and f2d in the notebook. The gd algorithm update on the each step to a more optimal solution, using the iterative formula: xk = xk−1 − λ∇f(xk−1), where x is the point to update, k the iteration, lambda the learning/update rate. The stopping condition is when the maximum number of iteration is reached or norm of (xk-xk-1) is smaller than a specific value, in this case: 1e-6.

In the 1D case, f(x)= x^2 - x + 1. Initial value is chosen to be -4 and lambda is set at 0.1. This takes 61 steps to reach the minumum x at 0.499995586412842. This is arbitrarily closed to the minumumn at 0.5 calculated analytically. See 1D_GD_update_history_graph.png for plotting and update details.

In the 2d case, f(x,y)= x^2 + y^2. Initial value is chosen to be (-4,4). and lambda is set at 0.1. This takes 63 steps to reach the minumum x at (-3.13855086769334e-6, -3.13855086769334e-6). This is arbitrarily closed to the minimum at (0,0) calculated analytically. See 1D_GD_update_history_graph.png for plotting and update details.

For more details on implementation, see HW02_q2.ipynb of the section "Homework 2: Iterative Reconstruction". Gradient descent algorithm is implemented in the file - GradientDescent.py.

###  Homework 3:

This section is predominately to reconstruct an image x from the fx1orward operator (radon transform) matrix, denoted as A. and the sinogram, denoted as b. The reconstructed image x is a vector of dimension (82880x1), from flattening the projection image (296x280). The sinogram is a vector of dimension (201880), from flattening the sinogram sliced_sinogram_100 of dimension (721x280). Hence, in the equation Ax=b, A must then have the dimension (201880 x 82880) for this operation to be valid. A is initialized using aomip XrayOperator as such: aomip.XrayOperator(size, [721], np.linspace(0, 360, new_projs_cols), size[0]*1, size[0]*0.5), where the size of the 1st arguement is (296x280). For details of tge parameters of the operator, see XrayOperator.py. After that, the sinogram is filtered using the Ram Lak filter, as will be shown subsequently that it provides better reconstruction result. All implementation and results in homework 3 can be be obtained in the file HW02_q1345.ipynb. Ax=b is formulated as a least square problem 0.5*|Ax-b|^2 and x is solved for using gd, with different regularization term added which will be discussesed next. After solving for x, the vector is then reshaped into the original image dimension (296x280) and this will be the reconstructed image.

From then on, A,x,b will always refers to the forward operator, reconstructed image and singoram respectively. lambda will refers to the learning rate, as it is stated as such in the homework sheet. This abbreviation will be used in the subsequent section, unless otherwise stated.

#### i) Least Squares

This is handled by the function gradientDescent and the update direction is the gradient of the least square function A'|Ax-b|. The parameters used are:  A,b, initial value x0: a vector of the same dimension as x with all elements set to -1, iteration = 5, lambda = 1e-3. The reconstructed image can be seen in reconstructed_gd_sino150.png. If b is unfiltered, the reconstructed image is reconstructed_gd_unfiltered_sino150.png. Here, I have to minimize the number of iteration to ensure that it can be run in a reasonable time for this particular experiment. Nevertheless, the reconstruction is still discernible.

#### ii) L2-Norm squared

This is handled by the function gradientDescentTikhonov and the update direction is the gradient of the least square function + Tikohonov regularization term: A'|Ax-b| + beta*|x|. The parameters used are:  A,b, initial value x0: a vector of the same dimension as x with all elements set to -1, iteration = 5, lambda = 1e-3 and regularization term beta = 2. The reconstructed image can be seen in reconstructed_tiknonov_sino150.png.

#### iii) Huber Functional

The derivative of  huber functional is dL_delta_x, implemeted as np.where(abs_x <= delta, x, delta * np.sign(x)). Formula is from the homework sheet [4].

This is handled by the function gradientDescentHuber and the update direction is the gradient of the least square function + Huber Functional regularization term: A'|Ax-b| + dL_delta_x. The parameters used are:  A,b, initial value x0: a vector of the same dimension as x with all elements set to -1, iteration = 5, lambda = 1e-3 and regularization term delta = 2. The reconstructed image can be seen in reconstructed_Huber_sino150.png.

#### iv) Fair potential

The derivative of Fair Potential is dPhi_delta_x, implemeted as x / (1 + (x/lambd)). Formula is from the homework sheet [4].

This is handled by the function gradientDescentFair and the update direction is the gradient of the least square function + Fair potential regularization term: A'|Ax-b| + dPhi_delta_x. The parameters used are:  A,b, initial value x0: a vector of the same dimension as x with all elements set to -1, iteration = 5, lambda = 1e-3 and regularization term lambd = 2. The reconstructed image can be seen in reconstructed_Fair_sino150.png. Note that lambda refers to learning rate and lamd is the Fair potential regularization parameter.

*** Note that the results of this section (reconstructed image) is obtained in my local machine, when I execute the exact same code on the remote recsrv03, recosntruction is just random noise.

###  Homework 4: Finite Differences

The forward difference method is used to compute L, the finite difference operator, using the method described in slide 26 of week 2 lecture slides[3] to obtain the matrix T. The problem can be formulated as 0.5*|Ax-b|^2 + 0.5*|Lx|^2. Everything is almost the same as in Homework 4, except the gradient update direction, as the 0.5*|Lx|^2 needs to be differentiated and added to the gradient term. The result of differentiating this term is simply |L'Lx| or in python term ((L.T).dot(L)).dot(x). Therefore, the gradient for Tikhonov regularization is now: A'|Ax-b| + beta*|L'Lx|; for Huber Functional is  A'|Ax-b| + |L'LdL_delta_x|; for Fair Potential is A'|Ax-b|+ |L'LdPhi_delta_x|. All other parametes such as lambda, initial value xo, regularization parameters followed exactly those set in Homework 3.

THe results are not desirable, as L'L is too large to be computed in my local environment, this steps have to be performed on a GPU. However, the reconstruction on GPU is bad as stated previously, nonetheless, the reconstructed images are reconstructed_FiniteDifference_sino150.png (reconstructed with Tikhonov regularization); reconstructed_Huber_FiniteDifference_sino150.png (reconstructed with Huber Functional as regularization); reconstructed_Fair_FiniteDifference_sino150.png (reconstructed with Fair Potential as regularization).


### Homework 5: Iterative algorithm vs FBP

In this part, I reconstruct the same x using the same A,b using filtered back backprojection (fbp)to compare the results between these 2 approaches. Again b is filtered before backprojection, and backprojection is performed by using the given function applyAdjoint in XrayOperator.py. The reconstructed image is reconstructed_fbp_sino150.png. Then, in the notebook HW02_q1345, see the section under Homwork 5, the results of all reconstruction methods are plotted side by side. At a first glance, the difference between all 5 reconstruction methods aren't too pronounced, the overall cross section of the seashell is discernible, perhaps only the color contrasts are different between all of them. Using fbp as a benchmark reconstruction algorithm to evaluate againt the gd iterative with different regularization parameters, the MSE error between fbp and various iterative algorithms is shown as such: error between fbp and reconstruction without regularization is 2560.7110759391308; error between fbp and reconstruction with Tikhonov regularization is  2591.3540618943857; error between fbp and reconstruction with Huber Functional regularization is 2560.70016373615; error between fbp and reconstruction with Fair Potential regularization is 2560.699798583447.

Next, random noise is added to the filtered sinogram to obtain the reconstruction result.Again all parameteres are set as in the previous sections. The reconstructed images are: reconstructed_gd_noisy.png; reconstructed_tiknonov_noisy.png; reconstructed_Huber_noisy.png; reconstructed_Fair_noisy.png. The MSE error between fbp and various iterative algorithms with noise added is shown as such: error between fbp and reconstruction without regularization is 2548.8305657898545; error between fbp and reconstruction with Tikhonov regularization is  2579.4322549440826; error between fbp and reconstruction with Huber Functional regularization is 2548.8196578739226; error between fbp and reconstruction with Fair Potential regularization is 2548.8192930941996. The results show that using Tikhonov as regularization gives a worse results, but without regularization and regularizing with Huber Functional and Potential regularization gives only very minor difference. Perhaps the problem is due to choice of regularization parameters. Also, I also investigate the time taken for each algorithm: fbp = 2.315293073654175; gd withoug regularization = 48.35488414764404; gd with Tikhonov regularization = 37.62824606895447; gd with wHuber Functional regularization = 48.92222285270691; gd with Fair Potential regularization = 49.4227180480957. Surprisingly gd with Tikhonov regularization takes the least time to compute but also has the worst reconstruction result.

Finally, I attempted the Helsinki tomographic challenge, reconstruction Phantom A, arc 360 and difficulty 1. Fbp is used to reconstructed the image, again ram lak filter is used to filter the sinogram. The reconstructed image is challenge_reconstruction.png. The reconstruction is submitted to the leaderboard and it achieves a score of 0.535065.

## -----Part 3: Challenges, limitations, lessons and future improvement ---------------------------------------

My main challenges are in homework 1,3,4. In homework 1, in the beginning the process of extracting a row out of projection image to construct the sinogram is very unintuitive to me. In homework 3, a long time is spent looking for a desirable learning rate, lambda. At first, the reconstruction obtained are all blank image, looking at the values of x it shows an extremly large numbers for all pixels, this compels me to keep experimenting with even lower lambda until a decent reconstruction is obtained. Moreover, I first do the experiment on my local environment and expect the remote server to produce at least a similar if not the exact same results. But I was shocked to find out the reconstruction results are very bad. This is the limitaion of this week's assignment and in the future testing should be done early on in the remote server.


## -----Part 4: Reference ---------------------------------------------------------------------------------------------

1. Kamutta, Emma, Mäkinen, Sofia, & Meaney, Alexander. (2022). Cone-Beam Computed Tomography Dataset of a Seashell (1.1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6983008
2. Meaney, Alexander, Silva de Moura, Fernando, & Siltanen, Samuli. (2022). Helsinki Tomography Challenge 2022 open tomographic dataset (HTC 2022) (1.2.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7418878
3. https://gitlab.lrz.de/groups/IP/teaching/applied-optimization-methods-for-inverse-problems/-/wikis/uploads/1b42972297742d1155a08c4c7cca304c/2023-05-02-01-applications.pdf
4. https://gitlab.lrz.de/groups/IP/teaching/applied-optimization-methods-for-inverse-problems/-/wikis/uploads/17cb5fb2edcacea3981405d3a27f5b9f/homework02.pdf