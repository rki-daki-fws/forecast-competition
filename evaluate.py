import pandas as pd
import numpy as np
import os
import sys
from datetime import date
from github import Github


if __name__ == "__main__":
    
    # open leaderboard
    lb = pd.read_csv("leaderboard.csv", header=0)
    print(lb)
    
    submissions_dir = "./submissions"
    # get all submission filenames
    submission_files = [f for f in os.listdir(submissions_dir) if os.path.isfile(os.path.join(submissions_dir, f))]
    print(submission_files)
    
    to_evaluate = []
    # check which submissions were not evaluated yet
    for submission in submission_files:
        team, model = submission[:-4].split("-")
        if not len(lb.loc[(lb["team"] == team) & (lb["model"] == model)]): # expects specific leaderboard columns
            to_evaluate.append(submission)
    
    if len(to_evaluate):
        # run evaluation
        gt = pd.read_csv("challenge-data/groundtruth.csv").to_numpy()
        
        lb_entries = lb.values.tolist()
        lb_columns = lb.columns.values.tolist()
        
        for f in to_evaluate:
            evaluation_file = os.path.join(submissions_dir, f)
            pred = pd.read_csv(evaluation_file).to_numpy()
            
            if gt.ndim != pred.ndim:
                print(f"Difference in dimensions to ground truth for prediction file {evaluation_file}")
                continue
                
            mse = np.square(gt[:, 1] - pred[:, 1]).mean()
        
            # add to leaderboard
            submission_date = date.fromtimestamp(os.path.getmtime(evaluation_file))
            team, model = f[:-4].split("-")
            lb_entries.append([submission_date, team, model, mse])
        
        new_lb = pd.DataFrame(lb_entries, columns=lb_columns)
        
        new_lb.sort_values("score", ascending=False, inplace=True)  # sort by score    
        print(new_lb)
        new_lb.to_csv("leaderboard.csv", index=False)  # update leaderboard file in repo
        
        # read leaderboard into str
        with open("leaderboard.csv", "r") as f:
            contents_new = f.read()
        
        
        # TODO commit leaderboard changes
        # check that only new results files were added
        print("Added token")
        token  = os.environ.get('GH_TOKEN')
        print(f"Token length: {len(token)}")

        g = Github(token)
        repo_name = os.environ.get('GITHUB_REPOSITORY')

        if repo_name is None:
            repo_name = 'mlbach/example-competition'

        repo = g.get_repo(repo_name)
        
        # update leaderboard file using github API
        contents_old = repo.get_contents("leaderboard.csv", ref="submit")
        repo.update_file(contents_old.path, "Update leaderboard with new submissions", contents_new, contents_old.sha, branch="submit")
        
        # TODO start pull request to main
        
        # try:
            # base = repo.get_branch("main")
            # head = repo.get_branch("submit")

            # merge_to_master = repo.merge("main",
                                # head.commit.sha, "Merge new submissions to master")

        # except Exception as ex:
            # sys.exit(ex)
    else:
        print("No new submissions to evaluate!")