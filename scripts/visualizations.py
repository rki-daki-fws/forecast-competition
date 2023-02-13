import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp
from scripts import utils


def drop_insufficient_levels(df, column, n_min=3):
    gr = df.groupby([column]).count()
    drop_levels = list(gr[gr.iloc[:, 0] < n_min].reset_index()[column])

    drop_idx = df[df[column].isin(drop_levels)].index
    return df.drop(index=drop_idx)


def create_boxplot(df, hue, var_name, title, ax, ylabel, color_palette, drop_columns=["refdate"], n_min=3):

    # drop rows of models with less than n_min submissions
    df = drop_insufficient_levels(df, "model", n_min)
    df = drop_insufficient_levels(df, "refdate", n_min)

    # drop columns
    if isinstance(drop_columns, list) and len(drop_columns):
        df = df.drop(drop_columns, axis=1)

    melted = pd.melt(df, id_vars=[hue], var_name=var_name)

    # then seaborn boxplot
    sns.boxplot(x=var_name, y='value', data=melted, hue=hue, ax=ax, palette=color_palette).set(title=title)
    ax.set_ylabel(ylabel)


def get_reference_date(filename):
    return filename.split(osp.sep)[-1].split("_")[0]


def replace_model_names(df, column):
    names = df[column].unique().tolist()
    for i, n in enumerate(names):
        df.replace(n, f"Model {i + 1}", inplace=True)
    return df


def plot_wis_compare_baseline(data, baseline, model, ax, xlim=None, title=""):

    mm = data.groupby(["target", "model"]).mean().reset_index()

    # plot model in question and baseline

    ax.plot(mm[mm.model == model].target, mm[mm.model == model].wis, c="blue", label=model, linewidth=1,
            marker="o", markersize=4)

    ax.plot(mm[mm.model == baseline].target, mm[mm.model == baseline].wis, c="green", label=f"baseline: {baseline}",
            linewidth=1, marker="o", markersize=4)

    # plot average over all models
    # mall = data.groupby(["target"]).mean().reset_index()
    # ax.plot(mall.target, mall.wis, c="blue", label="average score of all models", linewidth=1,
    #        marker="o", markersize=4)

    # make figure pretty
    ax.set_ylabel("Average WIS")
    ax.set_xlabel("Date")
    ax.tick_params(axis='x', labelrotation = 45)
    if xlim is not None:
        ax.set_xlim(xlim)
    if len(title):
        ax.set_title(title)


def plot_wis_over_time(data, baseline, ax, xlim=None, title=""):

    mm = data.groupby(["target", "model"]).mean().reset_index()

    # plot all models, highlighting baseline
    for m in mm.model.unique():
        col = "green" if m == baseline else "lightgrey"
        lab = f"baseline: {m}" if m == baseline else None
        ax.plot(mm[mm.model == m].target, mm[mm.model == m].wis, c=col, label=lab, linewidth=1,
                marker="o", markersize=4)
    # plot average over all models
    mall = data.groupby(["target"]).mean().reset_index()
    ax.plot(mall.target, mall.wis, c="blue", label="average score of all models", linewidth=1,
            marker="o", markersize=4)
    # make figure pretty
    ax.set_ylabel("Average WIS")
    ax.set_xlabel("Date")
    ax.tick_params(axis='x', labelrotation = 45)
    if xlim is not None:
        ax.set_xlim(xlim)
    if len(title):
        ax.set_title(title)
