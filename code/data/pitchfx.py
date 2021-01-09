import pandas as pd
import numpy as np
import datetime
import os
from sklearn.preprocessing import LabelBinarizer


def create_processed_data(
        path="/home/simon/Documents/601-Project/pitchfx2019/raw/",
        out_path="/home/simon/Documents/601-Project/pitchfx2019/processed/",
        year=2019,
        min_games=25
):
    # import files
    try:
        atbats = pd.read_csv(path + "atbats.csv")
        games = pd.read_csv(path + "games.csv")
        pitches = pd.read_csv(path + "pitches.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Data files not found. Please download them from Kaggle"
                                " and unzip it into " + str(path))
    # subset to year
    pitches_year = pitches[(pitches["ab_id"] // 1000000) == year]
    del pitches
    print("Imported year pitches:", pitches_year.shape)
    atbats_year = atbats[(atbats["ab_id"] // 1000000) == year]
    del atbats
    print("Imported year at bats:", atbats_year.shape)
    games_year = games[(games["g_id"] // 100000) == year]
    del games
    print("Imported year games:", games_year.shape)
    # patch umpires
    umpire_path = path + "umpires.csv"
    if os.path.isfile(umpire_path):
        umpires = pd.read_csv(umpire_path, index_col=0)
        umpires["date"] = pd.to_datetime(umpires["datetime"]).dt.strftime('%Y-%m-%d')

        abbrev = {
            'chc': 'chn',
            'cws': 'cha',
            'kc': 'kca',
            'laa': 'laa',
            'lad': 'lan',
            'nym': 'nyn',
            'nyy': 'nya',
            'sd': 'sdn',
            'sf': 'sfn',
            'stl': 'sln',
            'tb': 'tba',
        }

        umpires["away"].replace(abbrev, inplace=True)
        umpires["home"].replace(abbrev, inplace=True)

        umpires["home"].value_counts()
        games_year["home_team"].value_counts()
        umpires["away"].value_counts()
        games_year["away_team"].value_counts()

        umpires["unique_id"] = umpires["date"] + "_" + \
                               umpires["away"] + "-" + \
                               umpires["away_score"].astype(str) + "_" + \
                               umpires["home"] + "-" + \
                               umpires["home_score"].astype(str)
        games_year["unique_id"] = pd.to_datetime(games_year["date"]).dt.strftime('%Y-%m-%d') + "_" + \
                                  games_year["away_team"] + "-" + \
                                  games_year["away_final_score"].astype(int).astype(str) + "_" + \
                                  games_year["home_team"] + "-" + \
                                  games_year["home_final_score"].astype(int).astype(str)

        merged = pd.merge(left=games_year, right=umpires, how="inner", on="unique_id")
        hp_umpires = merged["hp_umpire"]
        games_year.loc[hp_umpires.index, "umpire_HP"] = hp_umpires.values
    print("Patching umpires:", games_year.shape)
    games_year = games_year[~games_year["umpire_HP"].isna()]
    print("With umpires:", games_year.shape)
    # Keep only called balls and strikes
    pitchfx = pitches_year[pitches_year["code"].isin(["B", "C"])]
    del pitches_year
    print("Keep only called balls and strikes:", pitchfx.shape)
    # join into pitches
    pitchfx = pd.merge(pitchfx, atbats_year, how="left", on="ab_id")
    del atbats_year
    print("Merged at bats into pitches:", pitchfx.shape)
    pitchfx = pd.merge(pitchfx, games_year, how="left", on="g_id")
    print("Merged games into pitches:", pitchfx.shape)
    # regular season only
    pitchfx["date"] = pd.to_datetime(pitchfx["date"], format='%Y-%m-%d')
    pitchfx = pitchfx[pitchfx["date"] <= datetime.datetime(year, 9, 29, 0, 0, 0)]
    pitchfx = pitchfx[pitchfx["date"] >= datetime.datetime(year, 3, 28, 0, 0, 0)]
    print("Keep only regular season games:", pitchfx.shape)
    # umpires with many games
    umpires_counts = games_year["umpire_HP"].value_counts()
    del games_year
    umpires = umpires_counts.index[umpires_counts >= min_games]
    print("Umpires with at least", min_games, "games:", umpires.shape)
    pitchfx = pitchfx[pitchfx["umpire_HP"].isin(umpires)]
    print("Subset to umpires with at least", min_games, "games:", pitchfx.shape)
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
    print("Writing to ", out_path + "pitchfx.csv")
    pitchfx.to_csv(out_path + "pitchfx.csv", index=False)
    print("Success.")


class PitchFxDataset:

    def __init__(self, path="./data/pitchfx/", force=False, x_lim=10.5 / 12):
        self.pitchfx = None
        self.load_pitchfx(force, path)
        self._sz_x_lim = x_lim
        self._set_correct_call()
        self._set_score_diff()
        self._standardize_pz()
        self._standardize_px()
        self.label_encoder_ = LabelBinarizer(neg_label=0, pos_label=1).fit(["B", "S"])
        self.pitchfx["type_01"] = self.label_encoder_.transform(self.pitchfx["type"])

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

    def _set_correct_call(self):
        self.pitchfx["type_from_sz"] = np.where(
            (self.pitchfx["px"] >= -self._sz_x_lim) &
            (self.pitchfx["px"] <= self._sz_x_lim) &
            (self.pitchfx["pz"] >= self.pitchfx["sz_bot"]) &
            (self.pitchfx["pz"] <= self.pitchfx["sz_top"]),
            "S",
            "B"
        )

    def _set_score_diff(self):
        self.pitchfx["score_diff_b_p"] = self.pitchfx["b_score"] - self.pitchfx["p_score"]

    def _standardize_pz(self):
        y = self.pitchfx[["sz_bot", "sz_top"]].mean()
        x_mean = self.pitchfx[["sz_bot", "sz_top"]].mean(axis=1)
        y_diff = y.diff()["sz_top"]
        x_diff = self.pitchfx["sz_top"] - self.pitchfx["sz_bot"]
        beta = y_diff / x_diff
        alpha = y.mean() - beta * x_mean
        self.pitchfx["pz_std"] = alpha + beta * self.pitchfx["pz"]

    def _standardize_px(self):
        self.pitchfx["px_std"] = np.where(
            self.pitchfx["stand"] == "L",
            -self.pitchfx["px"],
            self.pitchfx["px"]
        )
        self.pitchfx["pfx_x_std"] = np.where(
            self.pitchfx["stand"] == "L",
            -self.pitchfx["pfx_x"],
            self.pitchfx["pfx_x"]
        )

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
                col = feature + "_binned"
                bins = sorted(bins)
                labels = [
                    feature + "_({},{}]".format(bins[i], bins[i + 1]) if i > 0 else
                    feature + "_[{},{}]".format(bins[i], bins[i + 1])
                    for i in range(len(bins) - 1)
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
