import pandas as pd
import numpy as np

gt = pd.read_csv("challenge-data/groundtruth.csv").to_numpy()
pred = pd.read_csv("challenge-results/dummy-test.csv").to_numpy()

mse = np.square(gt[:, 1] - pred[:, 1]).mean()

print("Evaluation results:")
print(mse)