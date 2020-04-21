import pandas as pd
import re

def add_header(table, title, n_cols):
    """Add a title to a table"""
    table = table.replace(
        "\\toprule",
        "\\toprule\n"
        f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{title}}}}}\\\\\\addlinespace\n"
    )
    return table

def change_fontsize(table, fontsize):
    table = table.replace(
        "\\centering",
        f"\\centering{fontsize}"
    )
    return table

def add_divider(table, after, n_cols, text):
    table = table.replace(
        after,
        f"{after}\n\\addlinespace\n"
        f"\\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{text}}}}}\\\\\n\\addlinespace\n"
    )
    return table