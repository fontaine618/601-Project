import sys
import matplotlib.pyplot as plt
from visualisation.decoding import DecodingApp


if __name__ == "__main__":
    sys.path.extend(['/home/simon/Documents/601-Project/code'])
    plt.style.use("seaborn")
    encoder_path = "./../data/models/encoding/PCA10_umpire_balls_strikes.txt"
    self = DecodingApp(encoder_path)
    self.launch(port=9093)