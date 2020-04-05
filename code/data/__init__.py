import pandas as pd
import datetime


def create_processed_data(path="./data/pitchfx/"):
    # import files
    try:
        atbats = pd.read_csv(path + "atbats.csv")
        games = pd.read_csv(path + "games.csv")
        pitches = pd.read_csv(path + "pitches.csv")
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
    print("Merged games into pitches:", pitchfx.shape)
    # regular season only
    pitchfx["date"] = pd.to_datetime(pitchfx["date"], format='%Y-%m-%d')
    pitchfx = pitchfx[pitchfx["date"] <= datetime.datetime(2018, 10, 1, 0, 0, 0)]
    pitchfx = pitchfx[pitchfx["date"] >= datetime.datetime(2018, 3, 29, 0, 0, 0)]
    print("Keep only regular season games:", pitchfx.shape)
    # umpires with many games
    umpires_counts = games2018["umpire_HP"].value_counts()
    del games2018
    umpires = umpires_counts.index[umpires_counts > 29]
    print("Umpires with at least 30 games:", umpires.shape)
    pitchfx = pitchfx[pitchfx["umpire_HP"].isin(umpires)]
    print("Subset to umpires with at least 30 games:", pitchfx.shape)
    # clean dataset (arbitrary by looking at histograms)
    ranges = {
        "px": (-5, 5),
        "pz": (-0, 6),
        "break_angle": (-100, 100),
        "sz_bot": (0, 3),
        "sz_top": (2, 5),
    }
    for col, r in ranges.items():
        pitchfx = pitchfx[pitchfx[col].between(*r)]
        print("Subset to {} in {}:".format(col, r), pitchfx.shape)

    # write to file
    print("Writing to ", path + "pitchfx.csv")
    pitchfx.to_csv(path + "pitchfx.csv", index=False)
    print("Success.")


class PitchFxDataset:

    def __init__(self, path="./data/pitchfx/", force=False):
        self.pitchfx = None
        self.load_pitchfx(force, path)
        self.standardize_pz()

    def load_pitchfx(self, force, path):
        # Import PitchF/x pitchfx from file.
        # If force or file not found, we create the file.
        if force:
            create_processed_data(path)
        try:
            self.pitchfx = pd.read_csv(path + "pitchfx.csv")
        except FileNotFoundError:
            create_processed_data(path)
            self.pitchfx = pd.read_csv(path + "pitchfx.csv")
        expected_shape = (178922, 66)
        if not self.pitchfx.shape == expected_shape:
            raise ImportError(
                "PitchF/x pitchfx imported, but pitchfx set has unexpected shape: \n" +
                "\texpected " + str(expected_shape) +
                ", found: " + str(self.pitchfx.shape)
            )

    def standardize_pz(self):
        y = self.pitchfx[["sz_bot", "sz_top"]].mean()
        x_mean = self.pitchfx[["sz_bot", "sz_top"]].mean(axis=1)
        y_diff = y.diff()["sz_top"]
        x_diff = self.pitchfx["sz_top"] - self.pitchfx["sz_bot"]
        beta = y_diff / x_diff
        alpha = y.mean() - beta * x_mean
        self.pitchfx["pz_std"] = alpha + beta * self.pitchfx["pz"]

    def group_by(self, **kwargs):
        request = kwargs
        no_match = set(request) - set(self.pitchfx.columns)
        if len(no_match) > 0:
            raise ValueError("Some arguemnts do not match data frame columns: " + str(no_match))
        # create new columns
        cols = []
        for feature, bins in request.items():
            if bins == "all":
                cols.append(feature)
            else:
                col = feature+"_binned"
                bins = sorted(bins)
                labels = [
                    feature+"_({},{}]".format(bins[i], bins[i+1]) if i > 0 else
                    feature+"_[{},{}]".format(bins[i], bins[i+1])
                    for i in range(len(bins)-1)
                ]
                self.pitchfx[col] = pd.cut(
                    x=self.pitchfx[feature],
                    bins=bins,
                    labels=labels,
                    include_lowest=True
                )
                cols.append(col)
        df = self.pitchfx.groupby(by=cols)
        counts = df.agg("count")["px"]
        print(counts.agg(["min", "max", "count", "mean", "median"]))
        return df
