import warnings
from train import *
from param import *

if __name__ == '__main__':
    args = parse_args()
    repeat = 10000
    warnings.filterwarnings("ignore")
    for i in range(repeat):
        # ******************5-cv训练代码******************
        averages = fold_valid(args)
        print(averages)

    print("finish")
