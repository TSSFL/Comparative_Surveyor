"""Generate the demo pre/post survey pair shipped in this folder.

Unlike a uniform random draw, this simulates a real training intervention: each
respondent has a latent ability, each question has its own difficulty, and the
intervention shifts ability by a per-question effect size. Most questions improve
clearly, two barely move and one drifts down, which is what genuine pre/post data
looks like and gives the significance test something honest to detect.

    python make_demo_data.py
"""

import numpy as np
import pandas as pd

RESPONSES = ["Strongly Disagree", "Disagree", "Somewhat Agree", "Agree", "Strongly Agree"]
N = 120
SEED = 2026

QUESTIONS = {
    "Q1": ("I can confidently explain place value", 0.95),
    "Q2": ("I can teach fractions using visual models", 1.10),
    "Q3": ("I use real-life examples in numeracy lessons", 0.80),
    "Q4": ("I can diagnose why a pupil made an error", 1.25),
    "Q5": ("I feel confident teaching multiplication", 0.70),
    "Q6": ("I can adapt a lesson for a struggling pupil", 1.05),
    "Q7": ("I use group work in numeracy lessons", 0.45),
    "Q8": ("I can assess numeracy without a written test", 0.90),
    "Q9": ("I know where pupils commonly go wrong", 1.15),
    "Q10": ("I have enough teaching aids available", -0.20),  # resourcing, untouched
    "Q11": ("I can explain a concept more than one way", 1.00),
    "Q12": ("I have time to prepare numeracy lessons", 0.05),  # workload, untouched
}


def to_likert(latent, difficulty):
    """Cut a latent continuous score into the five ordered categories."""
    cuts = np.array([-1.2, -0.4, 0.4, 1.2]) + difficulty
    return np.digitize(latent, cuts)


def main():
    rng = np.random.default_rng(SEED)
    ability = rng.normal(0, 1, N)          # respondent ability, stable across waves
    pre, post = {}, {}

    for q, (_, effect) in QUESTIONS.items():
        difficulty = rng.normal(0, 0.35)
        noise_pre = rng.normal(0, 0.55, N)
        noise_post = rng.normal(0, 0.55, N)
        pre[q] = [RESPONSES[i] for i in to_likert(ability + noise_pre, difficulty)]
        post[q] = [RESPONSES[i] for i in to_likert(ability + effect + noise_post, difficulty)]

    ids = [f"T{i:03d}" for i in range(1, N + 1)]
    pre_df = pd.DataFrame({"Respondent": ids, **pre})
    post_df = pd.DataFrame({"Respondent": ids, **post})

    for name, df in (("demo_pre_survey", pre_df), ("demo_post_survey", post_df)):
        df.to_csv(f"{name}.csv", index=False)
        df.to_excel(f"{name}.xlsx", index=False)
        print(f"wrote {name}.csv and {name}.xlsx  ({len(df)} rows)")

    key = pd.DataFrame(
        [{"Question": q, "Statement": text, "Effect size": eff}
         for q, (text, eff) in QUESTIONS.items()]
    )
    key.to_csv("demo_question_key.csv", index=False)
    print("wrote demo_question_key.csv")


if __name__ == "__main__":
    main()
