from tensorflow.examples.tutorials.mnist import input_data
import warnings
warnings.filterwarnings("ignore")

def mnist():

    mnist = input_data.read_data_sets('data', reshape=False)

    return mnist

