import pandas as pd
import numpy as np
from github import Github
import sys
import re
import json
import os

def check_format(gt, pred):
    """
    Verify that submitted prediction is of same format as ground truth data.
    inputs: gt and pred are numpy ndarrays
    return: bool, whether or not format requirements are met
    """
    if not gt.ndim() == df.ndim():
        return False
        
    # TODO expand on other checks once ground truth data is chosen
        
    return True


if __name__ == "__main__":
    # check that only new results files were added
    print("Added token")
    token  = os.environ.get('GH_TOKEN')
    print(f"Token length: {len(token)}")
    #imgbb_token = os.environ.get('IMGBB_TOKEN')

    g = Github(token)
    repo_name = os.environ.get('GITHUB_REPOSITORY')

    if repo_name is None:
        repo_name = 'mlbach/example-competition'

    repo = g.get_repo(repo_name)

    print(f"Github repository: {repo_name}")
    print(f"Github event name: {os.environ.get('GITHUB_EVENT_NAME')}")

    event = json.load(open(os.environ.get('GITHUB_EVENT_PATH')))

    pr = None
    files_added = []
    files_changed = []
    file_pattern = re.compile(r"^submissions/([a-zA-Z0-9]+)-([a-zA-Z0-9]+)\.csv")

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
     
     
    # check that naming convention was adhered to
    for f in files_added:
        if file_pattern.match(f.filename) is None:
            sys.exit("Exiting automatic pipeline, submitted files did not adhere to naming convenction")
    
    #if pr is not None:
    #    pr.add_to_labels('other-files-updated')
    

    # check that format requirements are met by all submission files 
    gt = pd.read_csv("challenge-data/groundtruth.csv", sep=",", decimal=".", header=0).to_numpy()
    # TODO are files actually available here? or do we need to download them first?
    
    # probably because they are still in fork, and we only checked out branhc of base repo
    for f in files_added:
        submission = pd.read_csv(f.filename, sep=",", decimal=".", header=0).to_numpy()
        # check format requirements here
        if not check_format(gt, submission):
            sys.exit("Exiting CI pipeline, at least one submission file is not of required format")
            
