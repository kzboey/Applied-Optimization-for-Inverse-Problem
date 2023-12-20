# Homework

** This file serves as general guideline on how the homework is organized, and also the collection of reference I have used for this week's homework assignment. Specific documentation of the methods are results are in the aomip_hw1.ipynb to ease reading because plots are numerical results can be shown more easily and clearly on the .ipynb script.

Homework1:

The environment is setup on a macOS Catalina v10.15.7. Poetry is used to create the virtual environement. Then the steps stated 
in the gitlab page is followed: https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-boey-kai-zhe without any trouble.

Homework 2:

My main reference are chapters 5, 6 from the book [1] and Wikipedia page [2] to learn more about the filtered back projection algorithm. Besides, I also read about the details regarding the dataset from the paper [3], [4], more details regarding the specific dataset and methods to obtain them are discussed in following parts in homework 3 and homework 4.

Homework 3:

The dataset "Cone-Beam X-Ray CT Data Collection Designed for Machine Learning: Samples 1-8" from [5] is used for this part of the hoemwork. Specifically I am using samples 2. Just donwload Walnut2.zip from [5] and put it inside the hw01 folder, then everything is good go as the path and files to read are already configured in aomip_hw1.ipynb. The reason this dataset is chosen over others is because it has pre fat field corrected projection images. Hence, it contains the the dark field measurement data - di000000.tif and flat field meaurement data - – io000000.tif and – io000001.tif. Therefore, the contrast between and pre and post flat field corrected projection can be shown in part i) of homework 3. Data preparation for this dataset is reference from the author's implementation in their github page [6]. This dataset employs a cone beam setup with 1201 projections taken during a continuos, full rotation of the sample.

The implementation in homework is spread across 2 files, which are aomip_hw1.ipynb and preprocessing.py. The former handles the
data preparation, function call to handle the preprocessing, illustrations and textual explanation of the methods employed in thsi homework. While the latter handles the computational methods employed in the preprocessing steps which are the 6 methods stated in the homework sheet. In preprocessing.py, sections i) to vi) are labelled with comments to seperate them and to make code reading easier.

Homework 4: 

In this part, I directly use the the sinogram data from [7]. I choose this dataset because the sinogram data is already present. I tried to obtain the sinogram data using the same dataset from homework 3 and using the radon function in XrayOpereator.py, but the sinogram result is bad, probabily due to bad parameter tuning. However, this serves as an ongoing effort to learn more about radon transform. The filtered back projection algorithm is adapted from the implementation of the iradon function in scikit-image [8].

 

Reference: 
1. Computed Tomography: Algorithms, Insight, and Just Enough Theory, by Per Christian Hansen Jakob Sauer Jørgensen, and William R. B. Lionheart (2021)
2. https://en.wikipedia.org/wiki/Tomographic_reconstruction
3. Der Sarkissian, Henri, Lucka, Felix, van Eijnatten, Maureen, Colacicco, Giulia, Coban, Sophia Bethany, & Batenburg, K. Joost. (2019). Cone-Beam X-Ray CT Data Collection Designed for Machine Learning: Samples 1-8 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2686726 
4. Hämäläinen, Keijo, Harhanen, Lauri, Kallonen, Aki, Kujanpää, Antti, Niemi, Esa, & Siltanen, Samuli. (2015). Tomographic X-ray data of a walnut (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1254206
5. https://zenodo.org/record/2686726#.ZE9gjuxBy3I
6. https://github.com/cicwi/WalnutReconstructionCodes
7. https://zenodo.org/record/1254206#.ZE99l-xBy3I
8. https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/radon_transform.py