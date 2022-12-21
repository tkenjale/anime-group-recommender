This is an implementation of the group recommender algorithms described by Ortega et al. (2016) a the paper titled ["Recommending items to group of users using Matrix Factorization based Collaborative Filtering"](https://www.sciencedirect.com/science/article/pii/S0020025516300196). The base MF algorithms was inspired by [ExplicitMF](https://www.ethanrosenthal.com/2016/01/09/explicit-matrix-factorization-sgd-als/).

# How to use

1. Put the trained MF model in the data folder (for example "../../data/model_sgd_mf_v4_50__1666837325.pkl"), same thing for anime_encoder.csv, and group rating file ("../../data/group_rating_real.csv"). 
2. Run group_mf_development.ipynb

# Approach

We use a 2 steps approach:

1. Step 1: Offline training:
- Train the model on the entire dataset to get a set of item embeddings. We use stochastic gradient descent to train a MF model with bias term (described in the paper).
- Each anime is represented by a 128-dimensional vector.

2. Step 2: Online calculation:
- Applying ridge regression (described in the paper) to find the embedding of the virtual user.
- Calculate predicted rating of the virtual user, filter to get the un-watched animes and return the top 10 recommendation.

# 2. Phase 2 planning

Taking in the feedback of the midterm report, we are planning on improving our algorithms so that we'd get better score for the final.

Mid-term report feedbacks:

```
The report has no mention of the visualization approaches the team plans to implement. It was also not clear from the report the innovations the project is proposing. But I have awarded points for now. Please include more specific innovations in your final report. Also, please ensure you have made good progress on the visualization part in your final report. Also, for the final report, I strongly recommend using latex with two column format. Your report will look more structured and you can include more content in your report within the allowed limit on the number of pages. All the best!
```

```
There was very little description about your team's planned visualizations. Please include more details on this in the final report. It was also unclear what the innovations were in this project. The only innovation I picked up on reading the report was the MF based method, and even then it was not clear if you were using a novel method or the MF methods described in your literature survey. The description and work done on the algorithmic side. however, was well-written and it was great to see your progress in this area.
```

Notes about the final reports:
- More specific innovations in your final report (1. Algorithm, 2. Visualization)
- Format: Using latex with two-column format
- Regarding algorithmic innovations:
  - Experiments with different social functions
  - Experiments with before and after factorization
  - Clear evaluation methods and meaningful visualization of the experment results

## 2.1. Evaluation planning

`Clear evaluation methods and meaningful visualization of the experment results` -> We need to elaborate on this point by answering the following questions:
- What is/are the evaluation metric(s)? How would we calculate that from the real data (i.e. train/test split)
  - 
- What are the hyper parameters?
  - BF vs AF
  - Social functions:
    - Mean | AF & BF
    - Most happiness (Max) | AF & BF
    - Least misery (Min) | AF & BF
    - Borda Count | only AF
    - Average without Misery | only AF
  - Regularization factors (lambda)
- What are the charts that we want to plot?
  - 