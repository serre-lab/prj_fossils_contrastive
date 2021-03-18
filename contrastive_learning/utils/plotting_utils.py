"""
plotting_utils.py

Created by: Jacob A Rose
Created On: Tuesday, March 16th, 2021

Contains:

Functions useful for finely controlling the display of dataframes in a Jupyter Notebook by manually specifying CSS Styling.
Currently (Tuesday March 16th, 2021) not guaranteed to work right out of the box.
func magnify() -> List[Dict[str,Tuple]]:
func display_classification_report(report: pd.DataFrame, display_widget=False)

"""








from IPython.html import widgets
import pandas as pd
import seaborn as sns
from typing import List, Dict, Tuple


def magnify() -> List[Dict[str,Tuple]]:
    return [dict(selector="th",
                 props=[("font-size", "16pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "18pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '500px'),
                        ('font-size', '16pt')])
]

def display_classification_report(report: pd.DataFrame, display_widget=False):
    h_neg=(0, 359, 1)
    h_pos=(0, 359)
    s=(0., 99.9)
    l=(0., 99.9)
    
    if display_widget:
        @widgets.interact
#         def f(h_neg=(0, 359, 1), h_pos=(0, 359), s=(0., 99.9), l=(0., 99.9)):
        def f(h_neg=h_neg, h_pos= h_pos, s=s, l=l):
            return report.style.background_gradient(
                 cmap=sns.palettes.diverging_palette(h_neg=h_neg, h_pos=h_pos, s=s, l=l,
                                                       as_cmap=True))\
                         .set_precision(2)\
                         .set_caption('Global summary metrics')\
                         .set_table_styles(magnify())
    else:
        return report.style.set_precision(2)\
                           .set_caption('Global summary metrics')