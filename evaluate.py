import pandas as pd
import numpy as np
import os
import sys
from datetime import date
from github import Github
import dataframe_image as dfi


def evaluate(ground_truth, prediction):
    mse = np.square(ground_truth[:, 1] - prediction[:, 1]).mean()
    return mse


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
    new_lb.sort_values("score", ascending=False, inplace=True)  # sort by score
    new_lb.to_csv("leaderboard.csv", index=False)  # update leaderboard file in repo
    dfi.export(new_lb.iloc[:10, :], "leaderboard_snapshot.png", table_conversion="matplotlib")


if __name__ == "__main__":

    repo = init_repo_obj('rki-daki-fws/forecast-competition')

    # open leaderboard
    lb = pd.read_csv("leaderboard.csv", header=0)
    
    submissions_dir = "./submissions"
    # TODO update to submissions/teamname/file structure
    # get all submission filenames
    submission_files = [f for f in os.listdir(submissions_dir) if os.path.isfile(os.path.join(submissions_dir, f))]
    
    to_evaluate = []
    # check which submissions were not evaluated yet
    for submission in submission_files:
        team, model = submission[:-4].split("-")
        # TODO update leaderboard columns
        if not len(lb.loc[(lb["team"] == team) & (lb["model"] == model)]): # expects specific leaderboard columns
            to_evaluate.append(submission)
    
    if len(to_evaluate):
        # run evaluation
        gt = pd.read_csv("challenge-data/groundtruth.csv").to_numpy()  # TODO look for gt matching submission file
        
        lb_entries = lb.values.tolist()
        lb_columns = lb.columns.values.tolist()
        
        for f in to_evaluate:
            evaluation_file = os.path.join(submissions_dir, f)
            pred = pd.read_csv(evaluation_file).to_numpy()  # TODO update parquet file
            # format has already been validated, we can trust it here.

            score = evaluate(gt, pred)
        
            # add to leaderboard
            submission_date = date.fromtimestamp(os.path.getmtime(evaluation_file))
            team, model = f[:-4].split("-")
            lb_entries.append([submission_date, team, model, score])

        update_leaderboard_locally(lb_entries, lb_columns)

        # update leaderboard file and snapshot using github API
        update_repo_file(repo, "leaderboard.csv", "submit", "r")
        update_repo_file(repo, "leaderboard_snapshot.png", "submit", "rb")
        
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
        print("No new submissions to evaluate!")