import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from operations import *
np.random.seed(0)


def main():
    x = np.random.randint(-100, 100, 10)
    print(relu_numpy(x))


if __name__ == '__main__':
    main()
