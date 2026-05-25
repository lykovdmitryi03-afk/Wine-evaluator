# ML Project: Wine Quality Classifier

## Goal
Build a machine learning project using **Pandas**, **NumPy**, and **PyTorch** that predicts whether a red wine is “good” based on chemical properties.

A wine is labeled:

- `1` = good wine if `quality >= 7`
- `0` = not good wine if `quality < 7`

## What you will practice

- Loading CSV data with Pandas
- Exploring class distribution
- Creating features and labels
- Splitting data with NumPy
- Standardizing numerical features with NumPy
- Building a neural network with PyTorch
- Training with mini-batches
- Handling class imbalance using `pos_weight`
- Evaluating accuracy, precision, recall, and F1-score
- Saving model predictions to CSV

## Files

- `wine_quality_pytorch.py` — complete project code
- `README.md` — project explanation

## How to run

From this folder:

```bash
python wine_quality_pytorch.py
```

The script downloads the red wine quality dataset automatically from the UCI Machine Learning Repository.

## Outputs

After running, you should get:

- `wine_quality_model.pt` — saved PyTorch model checkpoint
- `wine_quality_predictions.csv` — test-set predictions you can inspect with Pandas or Excel

## Suggested extensions

1. Turn it into a 6-class classification task predicting exact quality score.
2. Add plots for training loss and validation F1-score.
3. Try different architectures: more layers, fewer layers, different dropout.
4. Tune the decision threshold instead of always using `0.5`.
5. Build a small function where a user enters wine chemical values and gets a prediction.

## How to start

Just download [requirements.txt](requirements.txt) with command in terminal *pip install requirements.txt*
