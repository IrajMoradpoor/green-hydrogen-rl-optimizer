import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from rl_train_agent import train_model
from evaluation import evaluate

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Training GRID scenario...")
train_model("grid")

print("Training PPA scenario...")
train_model("ppa")

print("Evaluating GRID...")
evaluate("grid")

print("Evaluating PPA...")
evaluate("ppa")