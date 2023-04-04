import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts import utils, scoring
from scripts import visualizations
import seaborn as sns
from math import ceil
from typing import Optional

from scripts.visualizations import plot_wis_compare_baseline


def aggregate_forecast_week(df, metric, grouping_columns=["refdate", "model"]):
    """ Efficient pandas implementation to aggregate forecast values to n-week ahead """
    new_columns = ["all", "week1", "week2", "week3", "week4"]

    df = df.copy()
    df.refdate = df.refdate.astype(np.datetime64)
    df["all"] = df[metric]
    df["week1"] = np.nan
    df["week2"] = np.nan
    df["week3"] = np.nan
    df["week4"] = np.nan

    rows_week1 = (df.target < df.refdate + np.timedelta64(7, "D"))
    rows_week2 = (df.target >= df.refdate + np.timedelta64(7, "D")) & (df.target < df.refdate + np.timedelta64(14, "D"))
    rows_week3 = (df.target >= df.refdate + np.timedelta64(14, "D")) & (df.target < df.refdate + np.timedelta64(21, "D"))
    rows_week4 = (df.target >= df.refdate + np.timedelta64(21, "D")) & (df.target < df.refdate + np.timedelta64(28, "D"))

    df.loc[rows_week1, "week1"] = df.loc[rows_week1, metric]
    df.loc[rows_week2, "week2"] = df.loc[rows_week2, metric]
    df.loc[rows_week3, "week3"] = df.loc[rows_week3, metric]
    df.loc[rows_week4, "week4"] = df.loc[rows_week4, metric]
    todrop = [c for c in df.columns if c not in grouping_columns + new_columns]

    aggregated = df.drop(todrop, axis=1).groupby(grouping_columns).mean()

    return aggregated.reset_index()


def aggregate_location(df, grouping_columns=["refdate", "model", "location"]):
    return df.groupby(grouping_columns).mean().reset_index()


def geometric_mean(arr):
    """ Caclulate geometric mean of input array """
    assert isinstance(arr, list) or isinstance(arr, np.ndarray)

    return np.prod(arr) ** (1 / len(arr))


def relative_score(df, metric_column, baseline="RKIsurv2-arima"):
    """
    # TODO improve docstring
    for refdate in df
        for team.model in df
            for team-model in df
                # using overlap in indices (actually compute pairwise)
                theta_mm' = average_wis model_a / average_wis model_b
        theta_m = product_M ^ (1/M)  # geometric mean
        theta = theta_m / theta_baseline
    """
    # do pairwise comparison
    # scale by baseline
    models = df["model"].unique().tolist()
    refdates = df["refdate"].unique()

    assert baseline in models, "Please specify a different baseline model!"
    assert metric_column in list(df.columns), f"Please specify a different metric!"

    # put baseline in first position
    idx_first, idx_baseline = 0, models.index(baseline)
    models[idx_first], models[idx_baseline] = models[idx_baseline], models[idx_first]

    thetas = {m: [] for m in models}
    for rd in refdates:

        model_rows = dict()
        for m_a in models:

            if model_rows.get(m_a) is None:
                model_rows[m_a] = (df["refdate"] == rd) & (df["model"] == m_a)

            if not any(model_rows[m_a]):
                if m_a == baseline:  # with no baseline, we cannot scale
                    break
                else:
                    continue

            thetas_a = []
            for m_b in models:
                if m_b == m_a:
                    continue

                if model_rows.get(m_b) is None:
                    model_rows[m_b] = (df["refdate"] == rd) & (df["model"] == m_b)

                if any(model_rows[m_b]):
                    # account for pair-wise overlap of target, location
                    loc_intersect = set(df.loc[model_rows[m_a], "location"]).intersection(
                        set(df.loc[model_rows[m_b], "location"]))

                    target_intersect = set(df.loc[model_rows[m_a], "target"]).intersection(
                        set(df.loc[model_rows[m_b], "target"]))
                    # here we assume that each model predicts the all combinations of location & target
                    # (not necessarily true)
                    a_mean = df.loc[model_rows[m_a] &
                                    (df["location"].isin(loc_intersect)) &
                                    (df["target"].isin(target_intersect)), metric_column].mean()
                    b_mean = df.loc[model_rows[m_b] &
                                    (df["location"].isin(loc_intersect)) &
                                    (df["target"].isin(target_intersect)), metric_column].mean()
                    thetas_a.append(a_mean/b_mean)

            if not len(thetas_a):
                continue  # there was not one model matching levels

            theta = geometric_mean(thetas_a)

            if m_a == baseline:
                theta_baseline = theta
                theta = 1.0
            else:
                theta = theta / theta_baseline  # scale

            thetas[m_a].append(theta)

    return thetas


def generate_spatial_matrix(df, bl_map, metric_col="wis"):
    # TODO create docstring

    models = df.model.unique().tolist()
    mat = np.full([len(bl_map), len(models)], np.nan)

    for i, tup in enumerate(bl_map.items()):
        bl, lks = tup
        per_model = relative_score(df[(df.location.isin(lks))], metric_col)
        # returned dict does not contain all models!
        for model, values in per_model.items():
            if len(values):
                col_idx = models.index(model)
                mat[i, col_idx] = np.nanmean(values)  # list of values for different reference dates

    empty_columns = np.where(np.all(np.isnan(mat), axis=0))[0].tolist()
    df = pd.DataFrame(mat, columns=models, index=list(bl_map.keys()))
    if len(empty_columns):
        df.drop(columns=[models[ec] for ec in empty_columns], inplace=True)
    return df


def coverage_probability(df, alpha=0.05):
    """
    calculate coverage probability per model
    """
    ci = int((1 - alpha) * 100)  # name in percentage
    # column is called 'within_x'
    col = f"within_{ci}"

    cp = dict()
    models = df.model.unique().tolist()
    for m in models:
        cp[m] = df.loc[df.model == m, col].sum() / len(df.loc[df.model == m, col])

    return cp


def wis_boxplot(df, hue, var_name):

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    df_21 = df[df.refdate < np.datetime64("2022-01-01")]
    df_22 = df[df.refdate >= np.datetime64("2022-01-01")]

    levels = set(df_21[hue]).union(set(df_22[hue]))
    colors = sns.color_palette(n_colors=len(levels))
    palette = {level: color for level, color in zip(levels, colors)}
    visualizations.create_boxplot(df_21, hue, var_name, "Year 2021", axes[0], "WIS", palette, drop_columns=["refdate"])
    visualizations.create_boxplot(df_22, hue, var_name, "Year 2022", axes[1], "WIS", palette, drop_columns=["refdate"])


def corresponding_boxplot_table(df, score="wis"):
    agg = aggregate_forecast_week(df, score)

    ci = agg.drop(["refdate"], axis=1).groupby(["model"]).apply(scoring.confidence_interval, alpha=0.05)
    means = agg.drop(["refdate"], axis=1).groupby(["model"]).mean().reset_index()

    printable = []
    for i, m in enumerate(means.model):
        row = [m]
        for j, v in enumerate(means.iloc[i, 1:]):
            if np.isnan(ci[m][0][j]) or np.isnan(ci[m][1][j]):
                ci_str = "/"
            else:
                ci_str = f"{round(ci[m][0][j], 2)} - {round(ci[m][1][j], 2)}"
            row.append(f"{round(v, 2)} ({ci_str})")
        printable.append(row)

    return pd.DataFrame(printable, columns=["model", "complete score (CI)", "week 1 score (CI)", "week 2 score (CI)",
                                            "week 3 score (CI)", "week 4 score (CI)"])


def regional_heatmap(df):
    # heatmap of relative wis
    bls = pd.DataFrame([[2, "Hamburg"],
                       [7, "Rheinland-Pfalz"],
                       [5, "Nordrhein-Westfalen"],
                       [15, "Sachsen-Anhalt"],
                       [11, "Berlin"],
                       [6, "Hessen"],
                       [9, "Bayern"],
                       [16, "Thüringen"],
                       [8, "Baden-Württemberg"],
                       [14, "Sachsen"],
                       [13, "Mecklenburg-Vorpommern"],
                       [12, "Brandenburg"],
                       [1, "Schleswig-Holstein"],
                       [4, "Bremen"],
                       [3, "Niedersachsen"],
                       [10, "Saarland"]], columns=["BundeslandId", "Bundesland"]).sort_values(by="Bundesland",
                                                                                              ignore_index=True)

    districts = {id: bl for _, id, bl in bls.itertuples()}  # district id: district name
    dist_mapping = {d: [] for d in districts.values()}  # district to landkreis id

    for id, dist in districts.items():
        dist_mapping[dist] = df[(df.location / 1000).astype(np.uint8) == id].location.unique()

    spatial_mat = generate_spatial_matrix(df, dist_mapping, metric_col="wis")

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    sns.heatmap(spatial_mat, annot=True, cmap="crest", ax=ax)
    plt.tight_layout()


def figure_wis_composition(results: pd.DataFrame, title: str, min_date: Optional[str] = None,
                           max_date: Optional[str] = None,
                           drop_models: Optional[list] = None) -> plt.Figure:
    """
    TODO document
    """

    condition = results.target < results.refdate + np.timedelta64(14, "D")
    if min_date is not None:
        condition = np.logical_and(condition, results.refdate >= np.datetime64(min_date))  # TODO assure format of yyyy-mm-dd

    if max_date is not None:
        condition = np.logical_and(condition, results.refdate < np.datetime64(max_date))
    # TODO np.logical only works with two conditions!!
    twoweekdf = results.loc[condition,].groupby(["model", "refdate"]).mean().reset_index()

    if drop_models is not None and len(drop_models):
        #drop_models = ["HHI-seq2seq4096d14", "HHI-seq2seq4096d21", "HHI-seq2seq4096d28"]
        twoweekdf.drop(np.where(twoweekdf.model.isin(drop_models))[0], inplace=True)

    # expand to full vector space of model, refdate, fill with zeros
    df_cols = list(twoweekdf.columns)
    models = twoweekdf.model.unique()
    refdates = np.sort(twoweekdf.refdate.unique())
    dflike = []
    for m in models:
        for ref in refdates:
            dflike.append([m, ref] + [0] * (len(df_cols) - 2))

    expanded = pd.DataFrame(dflike, columns=df_cols).set_index(["model", "refdate"])
    indexed = twoweekdf.set_index(["model", "refdate"])

    expanded.loc[indexed.index, :] = indexed.loc[indexed.index, :]
    expanded.reset_index(inplace=True)
    expanded.refdate = expanded.refdate.astype(str)

    # dynamically set n rows & cols based on number of models
    grouped = expanded.groupby(["model"])
    ncols = 3
    nrows = ceil(len(grouped.groups) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))

    for g, ax in zip(grouped.groups.items(), axes.flat):
        expanded.iloc[g[1]].plot.bar(x="refdate", y=["dispersion", "underprediction", "overprediction"], stacked=True,
                                     ax=ax,
                                     legend=False, color=["deepskyblue", "limegreen", "red"],
                                     grid=False)

    # update y axis limits
    y_max = max([ax.get_ylim()[1] for ax in axes.flat])
    i = 1
    for g, ax in zip(grouped.groups.items(), axes.flat):
        if not ax.has_data():
            fig.delaxes(ax)
        else:
            ax.set_title(f"{g[0]}")
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Reference date")
            i += 1

    # remove empty plots
    unaccounted = len(axes.flat) - len(grouped.groups)
    for ua in range(1, unaccounted + 1):
        fig.delaxes(axes.flat[-ua])

    fig.suptitle(title)
    fig.tight_layout()
    #save_current("figures/wis_composition_22.png")
    return fig


def create_figures_baseline_comparison(df: pd.DataFrame, baseline: Optional[str] = "RKIsurv2-arima") -> None:
    models = df.model.unique().tolist()
    cond_1week_ahead = df.target < df.refdate + np.timedelta64(7, "D")
    cond_4week_ahead = [df.target >= df.refdate + np.timedelta64(21, "D"),
                        df.target < df.refdate + np.timedelta64(28, "D")]

    years = ["2021", "2022", "2023"]
    arr_year = np.datetime_as_string(df.refdate, "Y")

    dfs_year, xlims = [], []

    for i, y in enumerate(years):
        dfs_year.append((df.loc[(cond_1week_ahead) & (arr_year == y),],  # one-week ahead
                         df.loc[np.logical_and(*(cond_4week_ahead)) & (arr_year == y),]))  # four-weeks ahead

        xlims.append((dfs_year[i][0].target.min(), dfs_year[i][0].target.max()))

    for m in set(models):
        if m == baseline:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(15, 11), sharey="col")

        for ax, dfy, lim, y in zip(axes.T, dfs_year, xlims, years):
            plot_wis_compare_baseline(dfy[0], baseline, m, ax[0], lim,
                                      title=f"{y}\nAverage 1-week ahead weighted interval score")
            plot_wis_compare_baseline(dfy[1], baseline, m, ax[1], lim,
                                      title=f"{y}\nAverage 4-week ahead weighted interval score")

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/baseline_{m}.png")


def create_table3(df_results):
    """
    TODO document
    """
    rel_wis = relative_score(df_results, "wis")
    rel_mae = relative_score(df_results, "mae")
    cov50 = coverage_probability(df_results, 0.5)
    cov95 = coverage_probability(df_results, 0.05)

    tab3 = []
    for key, values in rel_wis.items():
        row = [key, len(values), np.array(values).mean(), np.array(rel_mae[key]).mean(), cov50[key], cov95[key]]
        tab3.append(row)

    return pd.DataFrame(tab3, columns=["model", "n", "rel_wis", "rel_mae", "50% coverage", "95% coverage"])


def generate_tabbed_content(content: dict) -> str:
    """
    Generate html code for tabbed content which display is toggled by javascript
    """
    buttons = ""
    tabs = ""
    for i, (key, value) in enumerate(content.items()):
        button_class = "tablinks" if i > 0 else "tablinks active"
        buttons += f'<button class="{button_class}" onclick="openTab(event)">{key}</button>'
        tab_display = '' if i > 0 else ' style="display:block;"'
        tabs += f'<div id="{key}" class="tabcontent"{tab_display}>{value}</div>'

    return f'<div class="tabgroup"><div class="tab">{buttons}</div>{tabs}</div>'


# import IPython.display inside notebook, run display(HTML(code))
css_tab_navigation = """
<style>
body {font-family: Arial;}

/* Style the tab */
.tab {
  overflow: hidden;
  background-color: white;
}

/* Style the buttons inside the tab */
.tab button {
  background: hsl(210,50%,50%);
  /*float: left;*/
  border: 1px solid #ccc;
  outline: none;
  cursor: pointer;
  padding: 14px 16px;
  transition: 0.3s;
  font-size: 15px;
  color:white;
}

/* Change background color of buttons on hover */
.tab button:hover {
  background: hsl(210,50%,40%); 
}

/* Create an active/current tablink class */
.tab button.active {
  background-color: #ffffff;
  color: black;
  border-bottom: none;
}

/* Style the tab content */
.tabcontent {
  display: none;
  padding: 6px 12px;
  border: 1px solid #ccc;
}
</style>"""

# import IPython.display inside notebook, run display(HTML(code))
js_tab_navigation = """
<script>
function openTab(evt) {
  var i, tabcontent, tablinks, tabname, tabgroup;

  tabgroup = evt.currentTarget.parentElement.parentElement;

  tabcontent = tabgroup.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }

  tablinks = tabgroup.getElementsByClassName("tablinks");

  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  tabname = evt.currentTarget.textContent;
  document.getElementById(tabname).style.display = "block";
  evt.currentTarget.className += " active";

}
</script>
"""

if __name__ == "__main__":
    import time

    results = utils.load_results("../results.pickle")
    results["model"] = results["team"] + "-" + results["model"]
    results.drop(["team"], axis=1, inplace=True)
    before = time.time()
    #rel_wis = relative_score(results[(results.target < results.refdate + np.timedelta64(7, "D"))], "wis")
    """
    rows_week1 = (df.target < df.refdate + np.timedelta64(7, "D"))
    rows_week2 = (df.target >= df.refdate + np.timedelta64(7, "D")) & (df.target < df.refdate + np.timedelta64(14, "D"))
    rows_week3 = (df.target >= df.refdate + np.timedelta64(14, "D")) & (df.target < df.refdate + np.timedelta64(21, "D"))
    rows_week4 = (df.target >= df.refdate + np.timedelta64(21, "D")) & (df.target < df.refdate + np.timedelta64(28, "D"))
    """

    #print(rel_wis)
    # relative_score(resutls, "mae")

    #visualizations.wis_boxplot(aggregate_forecast_week(results, "wis"), "model", "forecast_period")


    ##### heatmap of rel_wis for different BL
    """bls = pd.read_csv("bl.csv").sort_values(by="Bundesland", ignore_index=True)
    districts = {id: bl for _, id, bl in bls.itertuples()}  # district id: district name
    dist_mapping = {d: [] for d in districts.values()}  # district to landkreis id

    for id, dist in districts.items():
        dist_mapping[dist] = results[(results.location / 1000).astype(np.uint8) == id].location.unique()

    spatial_mat = generate_spatial_matrix(results, dist_mapping, metric_col="wis")
    
    after = time.time()
    print(f"this took {after-before} s")

    plt.figure()
    sns.heatmap(spatial_mat, annot=True, cmap="crest")
    plt.tight_layout()
    plt.show()
    """

    # PROBLEM: this looks completely garbage for 22 right now, as forecast periods are not adjacent.
    # huge gaps! probably should upload more truth data, make forecasts for all dates.
    # in the meantime, it is probably best not to display this/make it optional

    # WIS over time  # TODO this was intended to be weekly values, not daily
    create_figures_baseline_comparison(results, baseline="RKIsurv2-arima", plot_2022=True)
