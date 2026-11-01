# Student Performance Predictor 🎓

A Streamlit web app that predicts a student's exam marks from **study hours**, **level of understanding** and **health**, using multiple linear regression (scikit-learn). It also helps students plan their study: targets, subject topics, exam dates and daily progress.

🔗 **Live app:** _add your Streamlit link here_

## Features
- Multiple linear regression: marks from study hours + understanding + health
- What-if simulation chart and a chart of what drives the marks
- Daily study-hours target, animated progress ring, study streak and calendar
- Subjects and topics with tick boxes, plus exam dates with a countdown
- Mock test marks and study-hours graph
- Motivational quotes, live clock and dark mode

## Run on your computer
```
pip install -r requirements.txt
py train.py
py -m streamlit run app.py
```
(On Mac or Linux, use `python3` instead of `py`.)

## Files
| File | What it does |
|---|---|
| `app.py` | The website |
| `train.py` | Trains the regression model and saves `regressor.pickle` |
| `data.csv` | Sample data used to train the model |
| `requirements.txt` | Python libraries needed |
| `images/` | Pictures shown on the page |

## Data note
`data.csv` is **simulated sample data** (study hours, health 1-10, understanding 1-10 and marks). It is meant for learning and demonstration. Replace it with a real survey of students for real predictions.

## Team
- Name 1: role
- Name 2: role
- Name 3: role
