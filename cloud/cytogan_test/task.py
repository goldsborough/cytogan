from tensorflow.python.lib.io import file_io
import scipy.misc
import argparse

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--image', required=True)
parser.add_argument('--job-dir')
options = parser.parse_args()

if options.job_dir:
    print('job-dir: ', options.job_dir)

file = file_io.FileIO(options.image, mode='r')
image = scipy.misc.imread(file)
print(image.shape)
