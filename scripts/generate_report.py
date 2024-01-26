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

    return np.exp(np.log(arr).mean())  # overflow-proof


def geometric_mean(arr, axis=None):
    """
    Compute the geometric mean along the specified axis of a matrix. This is an overflow-proof implementation compared
    to np.prod(arr)**(1/len(arr)). Makes use of logarithmic rules.
    Parameters:
    - arr: numpy array
        The input matrix or vector.
    - axis: int or tuple of ints, optional
        Axis or axes along which the geometric mean is computed. Default is None,
        which means the geometric mean is computed over all elements of the matrix.
    Returns:
    - numpy array
        Geometric mean along the specified axis.
    """
    log_matrix = np.log(arr)
    mean_log = np.nanmean(log_matrix, axis=axis)
    return np.exp(mean_log)


def relative_score(df, metric_column, baseline="RKIsurv2-arima"):
    """
    Calculate the relative scores of each model with respect to a specified baseline.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing model predictions and relevant information.
    - metric_column (str): The column in the DataFrame representing the performance metric.
    - baseline (str, optional): The baseline model against which the relative scores are calculated.
                                Default is "RKIsurv2-arima".

    Returns:
    dict:
        A dictionary containing the relative scores of each model with respect to the specified baseline.
        The keys are model names, and the values are the corresponding relative scores.

    Raises:
    - AssertionError: If the specified baseline model or metric column is not found in the DataFrame.

    The function computes relative scores by comparing each model's performance against the baseline model.
    It uses a geometric mean approach to aggregate the pairwise ratios for each reference date.

    Example:
    ```python
    import pandas as pd

    # Sample usage
    data = {
        'refdate': ['2022-01-01', '2022-01-01', '2022-01-02', '2022-01-02'],
        'model': ['ModelA', 'ModelB', 'ModelA', 'ModelB'],
        'target': [1, 2, 1, 2],
        'location': ['CityX', 'CityY', 'CityX', 'CityY'],
        'metric': [0.8, 0.9, 1.2, 1.1]
    }

    df = pd.DataFrame(data)
    result = relative_score(df, metric_column='metric', baseline='ModelA')
    ```

    Note: The function assumes that the DataFrame structure follows the specified format with columns 'refdate', 'model',
    'target', 'location', and the specified metric_column.
    This implementation makes as much use of vectorization as possible. It is much faster than an analogeous
    implementation based on iterations, i.e.
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

    assert baseline in models, "Please specify a different baseline model!"
    assert metric_column in list(df.columns), f"Please specify a different metric!"

    merge_columns = ["refdate", "target", "location"]

    # outer join does the trick
    df = df.drop_duplicates()
    df_merged = df.copy()[["model"] + merge_columns + [metric_column]]
    for m in models:
        df_merged = pd.merge(df_merged, df[df.model == m][merge_columns + [metric_column]],
                             on=merge_columns, how="outer", suffixes=(None, f"_{m}"))
        # no duplicates to be deleted here

    # care for 'diagonal', set values of own model to nan
    for model in models:
        df_merged.loc[df_merged.model == model, f"{metric_column}_{model}"] = np.nan

    # most inner loop of OG function
    per_refdate_model = df_merged.groupby(["refdate", "model"]).mean()

    index_metric_column = per_refdate_model.columns.tolist().index(metric_column)
    per_refdate_model.iloc[:, index_metric_column+1:] = per_refdate_model.iloc[:, index_metric_column].values[:, np.newaxis] / per_refdate_model.iloc[:, index_metric_column+1:].values
    # calculate geometric mean
    per_refdate_model["theta"] = geometric_mean(per_refdate_model.iloc[:, index_metric_column+1:].values, axis=1)

    # scale by baseline
    for refdate in per_refdate_model.index.get_level_values('refdate').unique():
        try:
            baseline_value = per_refdate_model.loc[(refdate, baseline), "theta"]
        except KeyError:
            per_refdate_model.loc[refdate, "theta"] = np.nan
            continue  # no baseline for this refdate!
        per_refdate_model.loc[refdate, "theta"] = (per_refdate_model.loc[refdate, "theta"] / baseline_value).values

    theta_per_model = per_refdate_model.reset_index().groupby(["model"]).theta.mean()

    return theta_per_model.to_dict()


def generate_spatial_matrix(df, bl_map, metric_col="wis"):
    """
    Generate a spatial matrix based on relative scores for each model in a given DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing model evaluation results and location information.
    - bl_map (dict): A dictionary mapping baseline locations to lists of associated locations for spatial comparison.
    - metric_col (str, optional): The column in the DataFrame representing the performance metric.
                                  Default is "wis".

    Returns:
    pd.DataFrame:
        A spatial matrix where rows correspond to baseline locations, columns correspond to models, and the values
        represent relative scores for the specified metric.

    The function computes relative scores for the specified metric for each model across locations defined by the
    baseline mapping. The resulting spatial matrix provides a comprehensive overview of model performance in different
    geographical areas.

    Example:
    ```python
    import pandas as pd

    # Sample usage
    data = {
        'model': ['ModelA', 'ModelB', 'ModelA', 'ModelB'],
        'location': ['CityX', 'CityY', 'CityX', 'CityY'],
        'wis': [0.8, 0.9, 1.2, 1.1]
        # Add other necessary columns as per your actual DataFrame structure
    }

    baseline_map = {'CityX': ['CityY', 'CityZ'], 'CityY': ['CityX', 'CityZ']}
    df_results = pd.DataFrame(data)
    spatial_matrix = generate_spatial_matrix(df_results, baseline_map)
    ```

    Note: The function assumes that the input DataFrame structure follows the specified format with columns like
    'model', 'location', and the specified metric_col.
    """
    models = df.model.unique().tolist()
    mat = np.full([len(bl_map), len(models)], np.nan)

    for i, tup in enumerate(bl_map.items()):
        bl, lks = tup
        rel_score_per_model = relative_score(df[(df.location.isin(lks))], metric_col)

        for model, rel_score in rel_score_per_model.items():
            col_idx = models.index(model)
            mat[i, col_idx] = rel_score

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

    fig, axes = plt.subplots(1, 3, figsize=(16, 9))

    df_21 = df[df.refdate < np.datetime64("2022-01-01")]
    df_22 = df[(df.refdate >= np.datetime64("2022-01-01")) & (df.refdate < np.datetime64("2023-01-01"))]
    df_23 = df[df.refdate >= np.datetime64("2023-01-01")]

    levels = set(df[hue])
    levels = sorted(list(levels), reverse=True)
    colors = sns.color_palette("Set1", n_colors=len(levels))
    palette = {level: color for level, color in zip(levels, colors)}
    visualizations.create_boxplot(df_21, hue, var_name, "Year 2021", axes[0], "WIS", palette, drop_columns=["refdate"])
    visualizations.create_boxplot(df_22, hue, var_name, "Year 2022", axes[1], "WIS", palette, drop_columns=["refdate"])
    visualizations.create_boxplot(df_23, hue, var_name, "Year 2023", axes[2], "WIS", palette, drop_columns=["refdate"])


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
        plt.savefig(f"docs/figures/baseline_{m}.png")


def create_table3(df_results):
    """
    Create a summary table (Table 3) based on the input DataFrame containing model evaluation results.

    Parameters:
    - df_results (pd.DataFrame): The DataFrame containing model evaluation results, including columns such as
      'model', 'refdate', 'wis', 'mae', and other necessary information.

    Returns:
    pd.DataFrame:
        A summary table (Table 3) with columns representing model names, number of submissions, relative scores
        for WIS and MAE, and coverage probabilities at 50% and 95%.

    The function internally calls the following helper functions:
    - relative_score: Computes relative scores for WIS and MAE against a predefined baseline model.
    - coverage_probability: Calculates coverage probabilities at specified confidence levels (50% and 95%).

    Example:
    ```python
    import pandas as pd

    # Sample usage
    data = {
        'refdate': ['2022-01-01', '2022-01-01', '2022-01-02', '2022-01-02'],
        'model': ['ModelA', 'ModelB', 'ModelA', 'ModelB'],
        'wis': [0.8, 0.9, 1.2, 1.1],
        'mae': [0.5, 0.6, 0.8, 0.7]
        # Add other necessary columns as per your actual DataFrame structure
    }

    df_results = pd.DataFrame(data)
    result_table = create_table3(df_results)
    ```

    Note: The function assumes that the input DataFrame structure follows the specified format with columns like
    'refdate', 'model', 'wis', 'mae', and any other required columns for computation.
    """
    rel_wis = relative_score(df_results, "wis")
    rel_mae = relative_score(df_results, "mae")
    cov50 = coverage_probability(df_results, 0.5)
    cov95 = coverage_probability(df_results, 0.05)
    # calculate submissions per model
    n_submissions = df_results.groupby('model').refdate.agg(lambda x: len(pd.unique(x))).to_dict()

    tab3 = []
    for model in rel_wis.keys():
        row = [model, n_submissions[model], rel_wis[model], rel_mae[model], cov50[model], cov95[model]]
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


def plot_coverage_probability(df, coverage_col="within_", temporal_col="refdate", coverage_levels=None):
    if coverage_levels is None:
        coverage_levels = [50, 80, 95]
    else:
        if not isinstance(coverage_levels, list) or not len(coverage_levels) or not isinstance(coverage_levels[0], int):
            raise ValueError("'coverage_levels' needs to be list of ints")

    df = visualizations.drop_insufficient_levels(df, "model", 3)

    models = df.model.unique().tolist()
    n_levels = len(coverage_levels)

    fig, axes = plt.subplots(len(models), n_levels, figsize=(5 * n_levels, 5 * len(models)), sharey=True)
    colors = plt.rcParams["axes.prop_cycle"]()

    if not isinstance(axes, np.ndarray):  # means there is just one level
        axes = np.array(axes)[None]  # expand array

    for axis, model in zip(axes, models):
        rows = df[df.model == model]
        c = next(colors)["color"]

        for ax, level in zip(axis, coverage_levels):
            # ax.scatter(rows[temporal_col], rows[f"{coverage_col}{level}"], marker="D")
            ax.plot(rows[temporal_col], rows[f"{coverage_col}{level}"], marker="D", color=c)
            ax.plot(rows[temporal_col], [level / 100] * len(rows), linestyle="dashed", color="grey")
            ax.set_title(f"{level}% PI")
            ax.tick_params(axis='x', labelrotation=45)
            ax.set_xlabel("Time")

        axis[0].annotate(model, xy=(0, 0.5), xytext=(-axis[0].yaxis.labelpad - 5, 0),
                         xycoords=axis[0].yaxis.label, textcoords='offset points',
                         size='large', ha='right', va='center')

        axis[0].set(ylim=(0, 1), ylabel="Coverage probability")

    fig.tight_layout()
    plt.savefig("docs/figures/coverage.png")


# import IPython.display inside notebook, run display(HTML(code))
css_tab_navigation = """
<style>
body {font-family: Arial;}
.jp-Cell{
    max-width:1280px;
    margin:0 auto;
}

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

    results = utils.load_results("../results/res.pickle")
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

    # WIS over time
    create_figures_baseline_comparison(results, baseline="RKIsurv2-arima")
