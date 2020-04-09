from visualisation.decoding import DecodingApp
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.extend(['/home/simon/Documents/601-Project/code'])


if __name__ == "__main__":

    encoder_path = "./../data/models/encoding/PCA10_umpire_balls_strikes.txt"

    with open(encoder_path, "rb") as f:
        encoder, embeddings, x_range, y_range = pickle.load(f)

    min = embeddings.min(0)
    max = embeddings.max(0)
    n_components = encoder.encoder.n_components
    plt.style.use("seaborn")
    app = DecodingApp(n_components, encoder, min, max, x_range, y_range)

    app.launch(port=9093)