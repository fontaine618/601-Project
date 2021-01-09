import pandas as pd
from datetime import datetime
import statsapi

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 500)

def get_game_per_year(year):
    schedule = statsapi.schedule(start_date='03/28/{}'.format(year), end_date='09/29/{}'.format(year))
    return [x["game_id"] for x in schedule], [x["game_date"] for x in schedule]


def get_boxscore_from_game_id(game_id):
    return statsapi.boxscore_data(game_id)


def get_hp_from_string(string):
    start = string.find("HP: ") + 4
    end = string.find(". 1B:")
    return string[start:end]


def get_item_by_key_in_boxscore(boxscore, key):
    for item in boxscore["gameBoxInfo"]:
        if item["label"] == key:
            return item["value"]
    return ""


def get_relevant_from_boxscore(boxscore):
    # umpire
    umpires = get_item_by_key_in_boxscore(boxscore, "Umpires")
    hp_umpire = get_hp_from_string(umpires)
    # venue
    venue = get_item_by_key_in_boxscore(boxscore, "Venue").strip(".")
    # away
    away = ""
    if "abbreviation" in boxscore["teamInfo"]["away"]:
        away = boxscore["teamInfo"]["away"]["abbreviation"].lower()
    away_score = boxscore["away"]["teamStats"]["batting"]["runs"]
    # home
    home = ""
    if "abbreviation" in boxscore["teamInfo"]["home"]:
        home = boxscore["teamInfo"]["home"]["abbreviation"].lower()
    home_score = boxscore["home"]["teamStats"]["batting"]["runs"]
    return away, home, away_score, home_score, venue, hp_umpire


game_ids, game_times = get_game_per_year(2019)

data = [
    [game_id, game_time, *get_relevant_from_boxscore(get_boxscore_from_game_id(game_id))]
    for game_id, game_time in zip(game_ids, game_times)
]


umpires_df = pd.DataFrame(data)
umpires_df.columns = [
    "game_id", "datetime", "away", "home", "away_score", "home_score", "venue", "hp_umpire"
]
umpires_df.to_csv("/home/simon/Documents/601-Project/pitchfx2019/raw/umpires.csv")
