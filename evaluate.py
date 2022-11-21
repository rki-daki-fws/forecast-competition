import pandas as pd
import numpy as np
import os
import sys
import datetime
from github import Github
import glob
import dataframe_image as dfi
from warnings import warn
from check_submission import Submission


def evaluate(ground_truth: pd.core.frame.DataFrame, submission: Submission):
    """
    Find the intersection of levels date and location in ground truth and prediction, then calculate root mean squared
    error (RMSE) as a simple metric of evaluation.
    gt: DataFrame with columns [target, location, value]
    submission: Submission object inclduing DataFrame with columns [model_date, target, location, sample_id, value]
    return: tuple of length 3:
        float,RMSE of predictions and ground truth
        int, number of days that were forecasted
        int, number of locations for which forecast was done
    """
    pred = submission.df
    gt = ground_truth
    if isinstance(pred.target[0], str):
        pred.target = pred.target.apply(datetime.datetime.strptime, args=['%Y-%m-%d']).dt.date
    elif isinstance(pred.target[0], datetime.datetime):
        pred.target = pred.target.apply(pd.Timestamp.date)

    if isinstance(gt.target[0], str):
        gt.target = gt.target.apply(datetime.datetime.strptime, args=['%Y-%m-%d']).dt.date
    elif isinstance(gt.target[0], datetime.datetime):
        gt.target = gt.target.apply(pd.Timestamp.date)
    # TODO add more dtype checks and conversions

    # select by value, perhaps data is not contiguous
    start_date, end_date = get_date_range(submission.reference_date)
    date_intersect = set(pred.target).intersection(set(gt[(start_date <= gt.target) & (gt.target <= end_date)].target))
    location_intersect = set(pred.location).intersection(set(gt.location))
    forecast_len = len(date_intersect)
    num_locations = len(location_intersect)

    if not forecast_len or not num_locations:
        print("Could not match prediction to ground truth data in time or location")
        return None, 0, 0

    reduced_gt = gt[(gt.target.isin(date_intersect)) & (gt.location.isin(location_intersect))]
    reduced_pred = pred[(pred.target.isin(date_intersect)) & (pred.location.isin(location_intersect))]

    # gt values repeated for all sample_ids
    merged_df = pd.merge(reduced_pred, reduced_gt, how="left", on=["target", "location"])

    # calculate simple metric
    rmse = np.sqrt(np.nanmean(np.square(merged_df["value_x"] - merged_df["value_y"])))  # deal with nan values in data

    return rmse, forecast_len, num_locations


def get_date_range(start_date_str, days=28):
    """
    Get datetime.date objects of start and end day in 28 day interval.
    start_date_str: string of start date, i.e. '2022-10-10'
    return 2tuple of datetime.date
    """
    start_date = pd.Timestamp.date(datetime.datetime.strptime(start_date_str, "%Y-%m-%d"))

    dtime = datetime.datetime.combine(start_date, datetime.time())
    end_date = dtime + datetime.timedelta(days=days - 1)  # first date already counts
    end_date = pd.Timestamp.date(end_date)

    return start_date, end_date


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


def update_leaderboard_locally(entries, columns):
    """
    Overwrite local files of leaderboard csv and snapshot image
    return: None """
    new_lb = pd.DataFrame(entries, columns=columns)
    new_lb.sort_values("score", ascending=True, inplace=True, ignore_index=True)  # sort by score
    new_lb.to_csv("leaderboard.csv", index=False)  # update leaderboard file in repo
    #dfi.export(new_lb.iloc[:10, :], "leaderboard_snapshot.png", table_conversion="matplotlib")
    # aggregate leaderboard
    scoreboard = aggregate_scoreboard(lb)
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


def update_readme_file(scoreboard_markdown):
    readmefile = "README.md"
    with open(readmefile, "r") as f:
        readme = f.read()

    readme = update_scoreboard_md(readme, scoreboard_markdown)

    with open("README.md", "w") as fw:
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


if __name__ == "__main__":

    repo = init_repo_obj('rki-daki-fws/forecast-competition')

    # open leaderboard
    lb = pd.read_csv("leaderboard.csv", header=0)
    
    submissions_dir = "submissions"
    # get all submission filenames
    submission_files = glob.glob(f"{submissions_dir}{os.path.sep}*{os.path.sep}*.parquet")
    # naming already validated, can trust it here
    
    to_evaluate = []
    # check which submissions were not evaluated yet
    for submission in submission_files:
        _, team, filename = submission.split(os.path.sep)
        date, model, location, variable = filename.split("_")
        variable = variable.split(".")[0]
        if not len(lb.loc[(lb["team"] == team) & (lb["model"] == model) & (lb["target_date"] == date) &
                          (lb["location_type"] == location) & (lb["variable"] == variable)]):  # expects specific leaderboard columns
            to_evaluate.append(submission)
    
    if len(to_evaluate):
        # run evaluation
        lb_entries = lb.values.tolist()
        num_entries_before = len(lb_entries)
        lb_columns = lb.columns.values.tolist()
        
        for f in to_evaluate:
            team, f_remaining = f.split(os.path.sep)[1:]
            date, model, location_type, pred_variable = f_remaining.split(".")[0].split("_")

            gt = pd.read_csv(
                os.path.join("challenge-data", "evaluation", f'2022-10-02_{location_type}_{pred_variable}.csv'))

            pred = pd.read_parquet(f)  # pd.read_csv(f).to_numpy()
            # format has already been validated, we can trust it here.

            submiss = Submission(f, date, location_type, pred_variable, pred)
            score, forecast_len, num_locations = evaluate(gt, submiss)
            if score is None:
                warn(f"Dates or locations of prediction file {f} don't match ground truth!")
                continue

            # add to leaderboard
            submission_date = datetime.date.fromtimestamp(os.path.getmtime(f))
            # TODO add forecast_len, num_locations as column
            lb_entries.append([submission_date, team, model, date, location_type, pred_variable, score])

        if len(lb_entries) > num_entries_before:
            update_leaderboard_locally(lb_entries, lb_columns)

            # update leaderboard file and snapshot using github API
            update_repo_file(repo, "leaderboard.csv", "submit", "r")
            update_repo_file(repo, "README.md", "submit", "r")
            #update_repo_file(repo, "leaderboard_snapshot.png", "submit", "rb")

            # merge changes to main branch
            # TODO open pull request
            #  merge PR
            # try:
            #     base = repo.get_branch("main")
            #     head = repo.get_branch("submit")
            #
            #     merge_to_master = repo.merge("main",
            #                         head.commit.sha, "Merge new submissions to master")
            #
            # except Exception as ex:
            #     sys.exit(ex)
        else:
            sys.exit("No submissions file could be evaluated. Failing pipeline")
    else:
        print("No new submissions to evaluate!")