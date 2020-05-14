import re
import demoji
import torch
from nltk.stem.snowball import SnowballStemmer
import json
import numpy as np
from model import LSTMClassifier

demoji.download_codes()
vocab2index_path = "./data/vocab2index.json"
words_path = "./data/words.json"
model_path = "./data/inference_model.pt"

embed_dim = 160
hidden_dim = 500
output_dim = 1
n_layers = 2
dropout = 0.5
padding_idx = 1
bidirectional = True
batch_first = False

# Free Heroku has a size limitation of 500MB.
# I have tried to limit the number of packages to not
# hit the limit.
STOP_WORDS = [
    "'d",
    "'ll",
    "'m",
    "'re",
    "'s",
    "'ve",
    "a",
    "about",
    "above",
    "across",
    "after",
    "afterwards",
    "again",
    "against",
    "all",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "always",
    "am",
    "among",
    "amongst",
    "amount",
    "an",
    "and",
    "another",
    "any",
    "anyhow",
    "anyone",
    "anything",
    "anyway",
    "anywhere",
    "are",
    "around",
    "as",
    "at",
    "back",
    "be",
    "became",
    "because",
    "become",
    "becomes",
    "becoming",
    "been",
    "before",
    "beforehand",
    "behind",
    "being",
    "below",
    "beside",
    "besides",
    "between",
    "beyond",
    "both",
    "bottom",
    "but",
    "by",
    "ca",
    "call",
    "can",
    "cannot",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "down",
    "due",
    "during",
    "each",
    "eight",
    "either",
    "eleven",
    "else",
    "elsewhere",
    "empty",
    "enough",
    "even",
    "ever",
    "every",
    "everyone",
    "everything",
    "everywhere",
    "except",
    "few",
    "fifteen",
    "fifty",
    "first",
    "five",
    "for",
    "former",
    "formerly",
    "forty",
    "four",
    "from",
    "front",
    "full",
    "further",
    "get",
    "give",
    "go",
    "had",
    "has",
    "have",
    "he",
    "hence",
    "her",
    "here",
    "hereafter",
    "hereby",
    "herein",
    "hereupon",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "however",
    "hundred",
    "i",
    "if",
    "in",
    "indeed",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "keep",
    "last",
    "latter",
    "latterly",
    "least",
    "less",
    "made",
    "make",
    "many",
    "may",
    "me",
    "meanwhile",
    "might",
    "mine",
    "more",
    "moreover",
    "most",
    "mostly",
    "move",
    "much",
    "must",
    "my",
    "myself",
    "n't",
    "name",
    "namely",
    "neither",
    "never",
    "nevertheless",
    "next",
    "nine",
    "no",
    "nobody",
    "none",
    "noone",
    "nor",
    "not",
    "nothing",
    "now",
    "nowhere",
    "n‘t",
    "n’t",
    "of",
    "off",
    "often",
    "on",
    "once",
    "one",
    "only",
    "onto",
    "or",
    "other",
    "others",
    "otherwise",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "part",
    "per",
    "perhaps",
    "please",
    "put",
    "quite",
    "rather",
    "re",
    "really",
    "regarding",
    "same",
    "say",
    "see",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "serious",
    "several",
    "she",
    "should",
    "show",
    "side",
    "since",
    "six",
    "sixty",
    "so",
    "some",
    "somehow",
    "someone",
    "something",
    "sometime",
    "sometimes",
    "somewhere",
    "still",
    "such",
    "take",
    "ten",
    "than",
    "that",
    "the",
    "their",
    "them",
    "themselves",
    "then",
    "thence",
    "there",
    "thereafter",
    "thereby",
    "therefore",
    "therein",
    "thereupon",
    "these",
    "they",
    "third",
    "this",
    "those",
    "though",
    "three",
    "through",
    "throughout",
    "thru",
    "thus",
    "to",
    "together",
    "too",
    "top",
    "toward",
    "towards",
    "twelve",
    "twenty",
    "two",
    "under",
    "unless",
    "until",
    "up",
    "upon",
    "us",
    "used",
    "using",
    "various",
    "very",
    "via",
    "was",
    "we",
    "well",
    "were",
    "what",
    "whatever",
    "when",
    "whence",
    "whenever",
    "where",
    "whereafter",
    "whereas",
    "whereby",
    "wherein",
    "whereupon",
    "wherever",
    "whether",
    "which",
    "while",
    "whither",
    "who",
    "whoever",
    "whole",
    "whom",
    "whose",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
    "yet",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "‘d",
    "‘ll",
    "‘m",
    "‘re",
    "‘s",
    "‘ve",
    "’d",
    "’ll",
    "’m",
    "’re",
    "’s",
    "’ve",
]


def clean_tweet(tweet):
    # Remove usernames, "RT" and Hash
    tweet = re.sub(r"(RT|[@*])(\w*)", " ", tweet)
    # Remove Hashtag
    tweet = re.sub(r"#(\w+)", " ", tweet)
    # Remove links in tweets
    tweet = re.sub(r"http\S+", " ", tweet)
    # We remove "#" and keep the tags
    tweet = re.sub(
        r"(\\n)|(\#)|(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])",
        "",
        tweet,
    )
    tweet = re.sub(r"(<br\s*/><br\s*/>)|(\-)|(\/)", " ", tweet)
    # convert to lower case
    tweet = re.sub(r"[^a-zA-Z0-9]", " ", tweet.lower())  # Convert to lower case
    # Tweets are usually full of emojis. We need to remove them.
    tweet = demoji.replace(tweet, repl="")
    # Stop words don't meaning to tweets. They can be removed
    tweet_words = tweet.split()
    clean_tweets = []
    for word in tweet_words:
        if word not in STOP_WORDS and len(word) > 1:
            clean_tweets.append(word.strip())

    tweet = " ".join(clean_tweets)

    return tweet


def stem_tweet(tweet):
    stemmer = SnowballStemmer(language="english")
    tokenized_tweets = []
    doc = tweet.split()  # Tokenize tweet
    for word in doc:
        word = stemmer.stem(word)  # Stem word
        tokenized_tweets.append(word)
    return tokenized_tweets


def encode_sentence(text, vocab2index, N=75):
    tokenized = stem_tweet(text)
    encoded = np.zeros(N, dtype=int)
    enc = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc))
    encoded[:length] = enc[:length]
    return encoded, length


def predict(tweet, model, vocab2index):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tweet = clean_tweet(tweet)
    twt, length = tweet_encoded = encode_sentence(tweet, vocab2index)
    twt = twt[:length]
    assert len(twt) == length
    length = [length]
    tensor = torch.LongTensor(twt).to(device)
    model = model.to(device)
    tensor = tensor.unsqueeze(1)
    tensor_length = torch.LongTensor(length)

    pred = model(tensor, tensor_length)
    pred = torch.round(pred).detach().numpy()
    if pred[0] == 0.0:
        return "NOT_SPAM"
    else:
        return "SPAM"


def get_prediction(tweet):
    vocab2index = json.load(open(vocab2index_path))
    words = json.load(open(words_path))
    input_dim = len(words)
    model = LSTMClassifier(
        input_dim,
        embed_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        padding_idx,
        batch_first,
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return predict(tweet, model, vocab2index)
