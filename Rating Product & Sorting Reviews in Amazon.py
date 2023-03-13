
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# One of the most important problems in e-commerce is the accurate calculation of post-sale ratings given to products. Solving this problem means providing more customer satisfaction for the e-commerce website, highlighting the product for sellers, and providing a seamless shopping experience for buyers.
# Another problem that arises is the accurate sorting of comments given to products. Misleading comments can directly affect the sale of the product, resulting in both financial loss and loss of customers. By solving these two fundamental problems, e-commerce sites and sellers can increase their sales, while customers can complete their purchasing journey smoothly.

###################################################
# Story of Dataset
###################################################

# This dataset containing Amazon product data includes various metadata along with pproduct categories. The product with the most reviews in the electronics category has user ratings and reviews.

# Variables:
# reviewerID: User ID
# asin: Product ID
# reviewerName: User name
# helpful: Helpful rating rate
# reviewText: Review
# overall: Product rating
# summary: Summary of review
# unixReviewTime: Time of review
# reviewTime: Time of review in RAW
# day_diff: The number of days elapsed since the review
# helpful_yes: The number of up votes of the review
# total_vote: Total number of votes of the review


import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


###################################################
# Reading data
###################################################
df = pd.read_csv("Week 4/measurement_problems/Case Studies/Rating Product&SortingReviewsinAmazon/amazon_review.csv")

###################################################
# Time-based Weighted Average Rating
###################################################
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

# <30 days  weight = 0.3
# 30<d<60   weight =  0.26
# 60<d<75   weight = 24
# 75<d      weight = 0.2

df.loc[df["day_diff"] <= 30, "overall"].mean()*0.3 + \
                      df.loc[(df["day_diff"] > 30) & (df["day_diff"] < 60), "overall"].mean()*0.26 + \
                      df.loc[(df["day_diff"] > 60) & (df["day_diff"]) <= 75, "overall"].mean()*0.24 + \
                      df.loc[(df["day_diff"]) > 75, "overall"].mean() * 0.2


df["helpful_no"]=  df["total_vote"]-df["helpful_yes"]

###################################################
# Up-Down Difference Score
###################################################

def up_down_diff(dataframe):
    return dataframe["helpful_yes"] - dataframe["helpful_no"]
df["up_down_diff_score"] = up_down_diff(df)

###################################################
# Up-Ratio Score
###################################################

def up_ratio(up, total_vote):
    if total_vote == 0:
        return 0
    return up / total_vote

df["up_ratio_score"] = df.apply(lambda x: up_ratio(x["helpful_yes"], x["total_vote"]), axis=1)

df.sort_values("up_down_diff_score", ascending=False).head(20)

###################################################
# Wilson Lower Bound Score
###################################################

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wlb_score"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis = 1)

##################################################
# Best 20 review as a result of the process
###################################################

df.sort_values("wlb_score", ascending=False).head(20)

