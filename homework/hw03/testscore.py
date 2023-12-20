import argparse
from challenge import utils
from os.path import join
from tifffile import imsave, imread

parser = argparse.ArgumentParser(description='')

parser.add_argument('arg1', type=str, help='')

args = parser.parse_args()
arg1_value = args.arg1

path = '/u/home/boeb/aomip-boey-kai-zhe/homework/hw03/img'
groundtruth_file = 'htc2022_01b_recon.tif'
groundtruth_path = join(path,groundtruth_file)
groundtruth = imread(groundtruth_path)

recon_path = join(path,arg1_value)
recon = imread(recon_path)
score=utils.calculate_score(recon,groundtruth)
print('score: ',score)

