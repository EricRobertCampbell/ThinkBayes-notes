import pandas as pd

gss = pd.read_csv("gss_bayes.csv", index_col=0)
print(gss.head())

banker = gss["indus10"] == 6870
print(banker.head())

print(banker.sum())
print(banker.mean())


def prob(A) -> float:
    """ Computes the probability of a proposition A """
    return A.mean()


print(prob(banker))
