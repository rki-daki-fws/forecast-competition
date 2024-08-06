import pandas as pd
import numpy as np
import os
import sys
from github import Github
import glob
from warnings import warn
import datetime
from check_submission import Submission
import scoring, utils


def evaluate(ground_truth: pd.core.frame.DataFrame, submission: Submission):
    """
    Find the intersection of levels date and location in ground truth and prediction, then calculate root mean squared
    error (RMSE) as a simple metric of evaluation.
    gt: DataFrame with columns [target, location, value]
    submission: Submission object inclduing DataFrame with columns [model_date, target, location, sample_id, value]
    return: tuple of length 3:
        float,RMSE of predictions and ground truth # TODO adapt docstr
    """
    pred = submission.df
    gt = ground_truth
    pred.target = utils.series2date(pred.target)   # TODO this takes too long!
    gt.target = utils.series2date(gt.target)

    # select by value, perhaps data is not contiguous
    start_date, end_date = utils.get_date_range(submission.reference_date)
    date_intersect = set(pred.target).intersection(set(gt[(start_date <= gt.target) & (gt.target <= end_date)].target))
    location_intersect = set(pred.location).intersection(set(gt.location))
    forecast_len = len(date_intersect)
    num_locations = len(location_intersect)

    if not forecast_len or not num_locations:
        print("Could not match prediction to ground truth data in time or location")
        return None, None

    reduced_gt = gt[(gt.target.isin(date_intersect)) & (gt.location.isin(location_intersect))]
    reduced_pred = pred[(pred.target.isin(date_intersect)) & (pred.location.isin(location_intersect))]

    # gt values repeated for all sample_ids
    merged_df = pd.merge(reduced_pred, reduced_gt, how="left", on=["target", "location"])

    # calculate metrics per day, location
    #rmse = np.sqrt(np.nanmean(np.square(merged_df["value_x"] - merged_df["value_y"])))  # deal with nan values in data
    rmse = scoring.rmse(merged_df)
    # TODO test that rows of rmse and wis line up!  THIS IS ACTUALLY QUESTIONABLE!!
    #  on the other hand, if same grouping levels are used, order is deterministic
    # wis with merged df?
    wis = scoring.weighted_interval_score(reduced_gt, reduced_pred)  # wis, dispersion, under-, overprediction
    # per group --> shape [len(set(targets)) * len(set(locations)), 4]

    mae = scoring.mae(merged_df)  # mean absolute error
    mda = scoring.mda(merged_df)  # mean directional error

    within_50 = scoring.within_PI(merged_df, alpha=0.5)
    within_80 = scoring.within_PI(merged_df, alpha=0.2)
    within_95 = scoring.within_PI(merged_df, alpha=0.05)
    # TODO join columns
    # or do this later on? actually we don't need to bloat up memory with big dataframe, we could do it only before
    # writing to disc!
    #list(zip(rmse, *wis))  -> list of 4 lists of len 11172
    # list(zip(*rmse.index.values.tolist())) --> list of 2 lists of len 11172
    indices = list(zip(*rmse.index.values.tolist()))
    scores = [rmse.values.tolist()] + wis + [mae.tolist(), mda.tolist(),
                                             within_50.tolist(), within_80.tolist(), within_95.tolist()]
    return indices, scores


def init_repo_obj(repo_name_fallback):
    token = os.environ.get('GH_TOKEN')

    g = Github(token)
    repo_name = os.environ.get('GITHUB_REPOSITORY')

    if repo_name is None:
        print("repository name not set in environment")
        repo_name = repo_name_fallback

    return g.get_repo(repo_name)


def update_repo_file(repo, filename, branch_name, io_mode="r"):
    """
    Use PyGithub functionality to update file on remote branch, equivalent to commit and push in one operation.
    repo: PyGithub repository object
    filename: str, name and path to file (same location locally and in remote repository
    branch_name: str, name of the target branch where we want to update the file
    io_mode: str, either "r" or "rb" for byte files
    return: None
    """
    assert io_mode in ["r", "rb"]  # resulting in either str or bytes obj

    with open(filename, io_mode) as fp:
        contents_disc = fp.read()

    contents_repo = repo.get_contents(filename, ref=branch_name)
    repo.update_file(contents_repo.path, f"Update {filename} directly on branch {branch_name}",
                     contents_disc, contents_repo.sha, branch=branch_name)


# TODO remove or adapt!
def update_leaderboard_locally(entries, columns):
    """
    Overwrite local files of leaderboard csv and snapshot image
    return: None """
    new_lb = pd.DataFrame(entries, columns=columns)
    new_lb.sort_values("score", ascending=True, inplace=True, ignore_index=True)  # sort by score
    new_lb.to_csv("leaderboard.csv", index=False)  # update leaderboard file in repo
    #dfi.export(new_lb.iloc[:10, :], "leaderboard_snapshot.png", table_conversion="matplotlib")
    # aggregate leaderboard
    scoreboard = aggregate_scoreboard(new_lb)
    update_readme_file(scoreboard.to_markdown(index=False, floatfmt=".2f"))


def insert_markdown(md_base, md_insert, search_str1, search_str2):
    start = md_base.find(search_str1) + len(search_str1)
    end = md_base.find(search_str2)
    print(start, end)
    md_base = md_base[:start] + "\n\n" + md_insert + "\n\n" + md_base[end:]

    return md_base


def update_scoreboard_md(base_md, scoreboard_md):
    return insert_markdown(base_md, scoreboard_md, '<div class="start_scoreboard"></div>',
                           '<div class="end_scoreboard"></div>')


# TODO remove or adapt
def update_readme_file(scoreboard_markdown):
    readmefile = "README.md"
    with open(readmefile, "r") as f:
        readme = f.read()

    readme = update_scoreboard_md(readme, scoreboard_markdown)

    with open("../README.md", "w") as fw:
        fw.write(readme)


def aggregate_scoreboard(lb):
    gr = lb.groupby(["team", "model"])
    mean = gr.mean().reset_index()
    std = gr.std().reset_index()
    count = gr.count().reset_index().drop(["submission_date", "target_date", "location_type", "variable"], axis=1)

    agg = mean
    agg["score_std"] = std.score
    agg["forecasts"] = count.score

    agg.sort_values(["forecasts", "score"], ascending=[False, True], inplace=True, ignore_index=True)  # sort by score
    # round score, std to 2 digits
    agg.score = agg.score.round(2)
    agg.score_std = agg.score_std.round(2)
    agg.rename(columns={"team": "Team", "model": "Model", "score": "Score (mean)", "score_std": "Score (std)",
                       "forecasts": "#Forecasts"},
              inplace=True)
    return agg


def create_submissions_dict_per_model(submission_files):
    # create dictionary of model_pred_var: [fps]
    # 'model name' needs to include prediction variable and location here, as there might be multiple combinations
    model_submissions = {}
    for submission_fp in submission_files:
        team, filename = submission_fp.split(os.path.sep)[-2:]
        model_name = filename[11:-8]  # 'model_location_variable'
        model_submissions[model_name] = model_submissions.get(model_name, []) + [submission_fp]
    return model_submissions


def load_model_results(model, res_dir, submissions_dir):
    res = utils.load_results(res_dir, submissions_dir, f"res_{model}*.csv")
    # scratch rows with values not older than n weeks from today (refdate) (there might be delay in reports)
    res = utils.filter_last_weeks(res, 5)
    return res


def check_model_submissions(results, submission_files):
    to_evaluate = []
    for submission in submission_files:
        team, filename = submission.split(os.path.sep)[-2:]
        date, model, location, variable = filename.split("_")
        variable = variable.split(".")[0]
        rows = (results["refdate"] == date) & (results["team"] == team) & (results["model"] == model) & \
               (results["location_type"] == location) & (results["pred_variable"] == variable)

        if not len(results.loc[rows]):
            to_evaluate.append(submission)
    return to_evaluate


def evaluate_model_submission(to_evaluate, results_df_columns):
    # run evaluation
    new_entries = pd.DataFrame([])

    for j, f in enumerate(to_evaluate):
        team, f_remaining = f.split(os.path.sep)[-2:]
        refdate, model, location_type, pred_variable = f_remaining.split(".")[0].split("_")

        if utils.smaller_date(refdate, "2022-08-14"):
            gt = pd.read_csv(os.path.join("../challenge-data",
                                          "evaluation", f'2022-10-02_{location_type}_{pred_variable}.csv'))
        else:
            # we like to get stable results, with delayed case reports being accounted for
            # so the more recent the better
            gt = utils.get_opendata(str(np.datetime64("today")), location_type, pred_variable)

        pred = utils.load_data(f)  # format has already been validated, we can trust it here.

        submiss = Submission(f, team, model, refdate, location_type, pred_variable, pred)

        indices, scores = evaluate(gt, submiss)
        # rather than one score, evaluate should return rows (one per day)
        if indices is None:
            warn(f"Dates or locations of prediction file {f} don't match ground truth!")
            continue

        # prepare df like list, append to existing entries
        dfbase = [[it] * len(indices[0]) for it in [refdate, team, model, location_type, pred_variable]]
        new_entries = pd.concat([new_entries,
                                 pd.DataFrame(list(zip(*dfbase + indices + scores)), columns=results_df_columns)
                                 ])
    return new_entries


def save_model_evaluation(results, new_evaluations):
    if new_evaluations.shape[0]:
        grouping_columns = ["team", "model", "location_type", "pred_variable"]

        results = pd.concat([results, new_evaluations], ignore_index=True)
        model_info = results.loc[0, grouping_columns].values.tolist()
        results = results.loc[:, [col for col in results.columns.to_list() if col not in grouping_columns]]

        utils.df_to_split_files(results, f"../results/res_{'_'.join(model_info)}.csv")


def main():
    # load results
    res_dir = "../results"
    submissions_dir = "../submissions"

    # get all submission filenames
    submission_files = glob.glob(f"{submissions_dir}{os.path.sep}*{os.path.sep}*.parquet")
    # naming already validated, can trust it here

    model_submissions = create_submissions_dict_per_model(submission_files)

    for i, (model, files) in enumerate(model_submissions.items()):
        print(f"evaluating model {i}/{len(model_submissions)}: {model}")
        # check which submissions of this model were not evaluated yet
        results_model = load_model_results(model, res_dir, submissions_dir)
        to_eval_model = check_model_submissions(results_model, files)

        if len(to_eval_model):
            new_evaluations_model = evaluate_model_submission(to_eval_model, results_model.columns.values.tolist())
            save_model_evaluation(results_model, new_evaluations_model)


if __name__ == "__main__":
    import time

    before = time.time()
    main()
    after = time.time()
    print(f"this took {after - before} s")
