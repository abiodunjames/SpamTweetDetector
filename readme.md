## The Problem

One significant academic study estimated that up to 15% of Twitter users were automated bot accounts. 

[@Hubofml](https://twitter.com/hubofml) bot produces its contents by listening to specific hashtags and re-broadcasting to a broader range of audiences that are interested in them. 

In the past, people have tricked the bot into broadcasting tweets that contrary to its subject matter by appending #machinelearning, #computervision to tweets. 

For example, people would tweet things "RT @hubofml Everything is F**cked! #machinelearning," and the bot would retweet it. 


## Solution
I used text-categorization to analyze the content of tweets before retweeting. The PyTorch model was trained on Google Colab using 20K tweets sourced directly from twitter. 




