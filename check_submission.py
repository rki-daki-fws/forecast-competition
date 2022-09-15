import github.PullRequest
import pandas as pd
import numpy as np
import github
import sys
import re
import json
import os
from datetime import date
from dataclasses import dataclass
from io import StringIO, BytesIO
import requests


@dataclass
class Submission:
    filepath: str
    reference_date: str
    location_type: str
    target_type: str
    df: pd.DataFrame


def check_format(df_gt, submission):
    """
    Verify that submitted prediction is of the required format.
    inputs:
    df_gt: ground truth pandas DataFrame
    submission: Submission object including filepath and loaded DataFrame
    return: bool, whether or not format requirements are met
    """

    # load data from file and validate contents
    pred = submission.df
    assert len(pred.values)
    
    required_columns = ['model_date', 'target', 'location', 'sample_id', 'value']
    assert all([r in pred.columns.tolist() for r in required_columns])

    # check that model_date & target are dates
    assert isinstance(pred.loc[0, "model_date"], date)
    assert isinstance(pred.loc[0, "target"], date)

    # check that value is either int or float, depending on variable type
    assert (pred.dtypes["value"].type is np.float_ or pred.dtypes["value"].type is np.int_)

    # TODO check that location are present in gt file

    # TODO check that each combintation of target & location has all realizations of sample_id
        
    return True


def load_csv(pr: github.PullRequest.PullRequest, filepath: str):
    """
    Dowloads file contents and reads into pd.DataFrame.
    Utilizes PyGithub functionality to get file contents as str from the merged branch.
    Also works on other files like txt etc.
    pr: PullRequest object, which includes repo object of PR base repository
    filepath: path to file which we want to download
    returns: pd.DataFrame
    """
    file_contents = pr.base.repo.get_contents(filepath, ref=f"refs/pull/{pr.number}/merge")
    with StringIO(file_contents.decoded_content.decode()) as io_obj:
        df = pd.read_csv(io_obj, sep=",", decimal=".", header=0)
    return df


def load_parquet(pr: github.PullRequest.PullRequest, filepath: str):
    """
    Dowloads file contents and reads into pd.DataFrame. Workaround for PyGithub functionality, as github API can't read
    file. Pieces together raw.githubusercontent URL of file from PR head branch, as file is not available physically
    at base.repo/refs/pull/pr_number/merge.
    Works on any type of file, but only necessary if file is not readable by github API.
    Caution: works only on public repositories!
    pr: PullRequest object, which includes repo object of PR head repository
    filepath: path to file which we want to download
    returns: pd.DataFrame
    """
    # TODO improve to streaming files in case of large file size
    head_repo = pr.head.repo
    head_branch = pr.head.raw_data["ref"]
    # not fixed on commit. theroretically, file can already be in altered stated compared to PR
    url = f'https://raw.githubusercontent.com/{head_repo.full_name}/refs/heads/{head_branch}/{filepath}'
    req = requests.get(url, allow_redirects=True)

    with BytesIO(req.content) as io_obj:
        df = pd.read_parquet(io_obj)
    return df


def load_submission_data(pr: github.PullRequest.PullRequest, filepath: str):
    """
    Depending on file ending, calls appropriate function to download file and read into DataFrame.
    pr: PullRequest object, which includes repo object of PR head repository
    filepath: path to file which we want to download
    returns: pd.DataFrame
    """
    file_ending = filepath.split(".")[-1].lower()
    if file_ending == "csv":
        return load_csv(pr, filepath)
    elif file_ending == "parquet":
        return load_parquet(pr, filepath)
    else:
        sys.exit(f"Unsupported file type:{file_ending}")


if __name__ == "__main__":
    # check that only new results files were added
    print("Added token")
    token  = os.environ.get('GH_TOKEN')
    print(f"Token length: {len(token)}")

    g = github.Github(token)
    repo_name = os.environ.get('GITHUB_REPOSITORY')

    if repo_name is None:
        repo_name = 'rki-daki-fws/forecast-competition'  # TODO move to global config file

    repo = g.get_repo(repo_name)

    print(f"Github repository: {repo_name}")
    print(f"Github event name: {os.environ.get('GITHUB_EVENT_NAME')}")

    event = json.load(open(os.environ.get('GITHUB_EVENT_PATH')))

    files_added = []
    files_changed = []
    # expects files in submissions/TeamName/Date_Modelname_locationtype_forcastvalue.parquet
    file_pattern = re.compile(r"^submissions/([a-zA-Z0-9]+)/([0-9]{4}-[0-9]{2}-[0-9]{2})_([a-zA-Z0-9]+)_(LK|BL)_(cases|rvalue)\.parquet")

    #if os.environ.get('GITHUB_EVENT_NAME') == 'pull_request_target' or local:
    # Fetch the  PR number from the event json
    pr_num = event['pull_request']['number']
    print(f"PR number: {pr_num}")

    # Use the Github API to fetch the Pullrequest Object. Refer to details here: https://pygithub.readthedocs.io/en/latest/github_objects/PullRequest.html 
    # pr is the Pullrequest object
    pr = repo.get_pull(pr_num)

    # fetch all files changed in this PR and add it to the files_changed list.
    files_added +=[f for f in pr.get_files() if f.status=="added"]
    files_changed +=[f for f in pr.get_files() if f.status!="added"]

    if len(files_changed):
        # TODO add comment to PR?    
        # exit pipeline
        sys.exit("Exiting automatic pipeline, repo files were changed")
        
    if not len(files_added):
        sys.exit("Exiting automatic pipeline, no new results were submitted")
    else:
        print(f"{len(files_added)} CSV files have been submitted")
     
    submissions = []
    # check that naming convention was adhered to
    for f in files_added:
        matched = file_pattern.match(f.filename)
        if matched is None:
            sys.exit(f"Exiting automatic pipeline, submitted file did not adhere to naming convenction: {f.filename}")
        else:
            submissions.append(Submission(f.filename, matched.groups()[1], matched.groups()[3], matched.groups()[4],
                                          pd.DataFrame()))
    else:
        print("Submission files adhere to naming convention")

    for s in submissions:
        print(s.filepath)
        # open groundtruth file that matches
        gt = pd.read_csv(f"challenge-data/evaluation/{s.reference_date}_{s.location_type}_{s.target_type}.csv",
                         sep=",", decimal=".", header=0)

        # for security reasons working on pr base branch, need to download file contents from merged branch here
        # https://github.com/orgs/community/discussions/25961
        s.df = load_submission_data(pr, s.filepath)

        # check format requirements here
        if not check_format(gt, s):
            sys.exit("Exiting CI pipeline, at least one submission file is not of required format")
    else:
        # add automerge label to PR as no further checks are needed
        pr.set_labels("automerge")
