import pandas as pd
import datetime


def create_processed_data(path="./pitchfx/pitchfx/"):
    # import files
    try:
        atbats = pd.read_csv(path + "atbats.csv")
        games = pd.read_csv(path + "games.csv")
        pitches = pd.read_csv(path + "pitches.csv")
        player_names = pd.read_csv(path + "player_names.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Data files not found. Please download them from Kaggle"
                                " and unzip it into " + str(path))
    # subset to 2018
    pitches2018 = pitches[(pitches["ab_id"] // 1000000) == 2018]
    del pitches
    print("Imported 2018 pitches:", pitches2018.shape)
    atbats2018 = atbats[(atbats["ab_id"] // 1000000) == 2018]
    del atbats
    print("Imported 2018 at bats:", atbats2018.shape)
    games2018 = games[(games["g_id"] // 100000) == 2018]
    del games
    print("Imported 2018 games:", games2018.shape)
    # Keep only called balls and strikes
    pitchfx = pitches2018[pitches2018["code"].isin(["B", "C"])]
    del pitches2018
    print("Keep only called balls and strikes:", pitchfx.shape)
    # join into pitches
    pitchfx = pd.merge(pitchfx, atbats2018, how="left", on="ab_id")
    del atbats2018
    print("Merged at bats into pitches:", pitchfx.shape)
    pitchfx = pd.merge(pitchfx, games2018, how="left", on="g_id")
    del games2018
    print("Merged games into pitches:", pitchfx.shape)
    # regular season only
    pitchfx["date"] = pd.to_datetime(pitchfx["date"], format='%Y-%m-%d')
    pitchfx = pitchfx[pitchfx["date"] <= datetime.datetime(2018, 10, 1, 0, 0, 0)]
    pitchfx = pitchfx[pitchfx["date"] >= datetime.datetime(2018, 3, 29, 0, 0, 0)]
    print("Keep only regular season games:", pitchfx.shape)
    # umpires with many games
    umpires_counts = games2018["umpire_HP"].value_counts()
    umpires = umpires_counts.index[umpires_counts > 29]
    print("Umpires with at least 30 games:", umpires.shape)
    pitchfx = pitchfx[pitchfx["umpire_HP"].isin(umpires)]
    print("Subset to umpires with at least 30 games:", pitchfx.shape)
    # clean dataset
    ranges = {
        "": (0, 1),
    }
    for col, r in ranges.items():
        pass
    # write to file
    print("Writing to ", path + "pitchfx.csv")
    pitchfx.to_csv(path + "pitchfx.csv", index=False)
    print("Success.")


class PitchFxDataset:

    def __init__(self, path="./pitchfx/pitchfx/", force=False):
        # Import PitchF/x pitchfx from file.
        # If force or file not found, we create the file.
        if force:
            create_processed_data(path)
        try:
            self.pitchfx = pd.read_csv(path + "pitchfx.csv")
        except FileNotFoundError:
            create_processed_data(path)
            self.pitchfx = pd.read_csv(path + "pitchfx.csv")
        expected_shape = (300000, 200)
        if not self.pitchfx.shape == expected_shape:
            raise ImportError(
                "PitchF/x pitchfx imported, but pitchfx set has unexpected shape: \n" +
                "\texpected " + str(expected_shape) +
                ", found: " + str(self.pitchfx.shape)
            )

