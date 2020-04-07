import sys
sys.path.extend(['/home/simon/Documents/601-Project/code'])
import pickle
from visualisation.decoding import DecondingApp

encoder_path = "./../data/models/encoding/PCA10_umpire_balls_strikes.txt"

with open(encoder_path, "rb") as f:
    encoder, embeddings = pickle.load(f)

app = DecondingApp(encoder, embeddings)

app.launch()