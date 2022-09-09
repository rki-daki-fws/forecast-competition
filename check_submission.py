import pandas as pd
import numpy as np
from github import Github
import sys
import re
import json
import os
from datetime import date
from dataclasses import dataclass
from io import StringIO


@dataclass
class Submission:
    filepath: str
    reference_date: str
    location_type: str
    target_type: str
    df: None


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


if __name__ == "__main__":
    # check that only new results files were added
    print("Added token")
    token  = os.environ.get('GH_TOKEN')
    print(f"Token length: {len(token)}")

    g = Github(token)
    repo_name = os.environ.get('GITHUB_REPOSITORY')

    if repo_name is None:
        repo_name = 'rki-daki-fws/forecast-competition'  # TODO move to global config file

    repo = g.get_repo(repo_name)

    print(f"Github repository: {repo_name}")
    print(f"Github event name: {os.environ.get('GITHUB_EVENT_NAME')}")

    event = json.load(open(os.environ.get('GITHUB_EVENT_PATH')))

    pr = None
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
        if matched:
            sys.exit("Exiting automatic pipeline, submitted files did not adhere to naming convenction")
        else:
            submissions.append(Submission(f.filename, matched.groups[1], matched.groups[3], matched.groups[4]))
    else:
        print("Submission files adhere to naming convention")
    #if pr is not None:
    #    pr.add_to_labels('other-files-updated')

    for s in submissions:
        print(s.filepath)
        # first verify that there is groundtruth file that matches
        #assert os.path.isfile(f"challenge-data/incidences_reff_{s.reference_date}.csv")
        gt = pd.read_csv(f"challenge-data/incidences_reff_{s.reference_date}.csv", sep=",", decimal=".",
                         header=0)

        # for security reasons working on pr base branch, need to download file contents from merged branch here
        # https://github.com/orgs/community/discussions/25961
        file_contents = repo.get_contents(s.filepath, ref=f"refs/pull{pr.number}/merge")
        with StringIO(file_contents.decoded_content.decode()) as io_obj:
            s.df = pd.read_parquet(io_obj)
            # could allow CSVs here
            # if submission.filepath.lower().split(".")[-1] == "csv":
            #   s.df = pd.read_csv(f.filename, sep=",", decimal=".", header=0)

        # check format requirements here
        if not check_format(gt, s):
            sys.exit("Exiting CI pipeline, at least one submission file is not of required format")
    else:
        # add automerge label to PR as no further checks are needed
        pr.set_labels("automerge")
