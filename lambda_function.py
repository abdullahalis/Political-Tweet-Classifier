import json
import joblib
import os
import re
import string
import nltk

from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

lemmatizer= nltk.stem.wordnet.WordNetLemmatizer()
stopwords= stopwords.words('english')

# Get model and tfidf
model = joblib.load(os.path.join(os.path.dirname(__file__), "model.pkl"))
tfidf = joblib.load(os.path.join(os.path.dirname(__file__), "tfidf.pkl"))


# Convert part of speech tag from nltk.pos_tag to word net compatible format
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN
    
def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    #handle url
    processed = re.sub(r'http\S+', '', text)

    #handle emojis
    processed = emoji_pattern.sub(r'', processed)

    #handle apostrophe 'so 
    processed = processed.replace("'s","")

    #handle other apostrophe
    processed = processed.replace("'","")

    #handle all other punctuation
    processed = re.sub(r'[' + string.punctuation + r']+', ' ', processed).strip()

    #handle lowercase
    processed = processed.lower()

    # tokenize text
    tokens = word_tokenize(processed)

    # makes dictionary mapping token to treebank tag
    treebank_tags = nltk.pos_tag(tokens)

    # hold lemma tokens
    lemma_tokens = []

    for word, tree_tag in treebank_tags:
        # get corresponding wordnet tag from treebank tag
        wordnet_tag = get_wordnet_pos(tree_tag)

        # get lemma version of word and append to list
        lemma_token = lemmatizer.lemmatize(word, pos=wordnet_tag)
        lemma_tokens.append(str(lemma_token))
    
    return " ".join(lemma_tokens)



def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])
        text = body.get("text", "")
        if not text:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No input text provided"})
            }
        
        # Process text and get prediction
        processed_text = process(text)

        X = tfidf.transform([processed_text])
        prediction = model.predict(X)[0]

        if prediction == 0:
            prediction = "Republican"
        else:
            prediction = "Democrat"

        return {
            "statusCode": 200,
            "body": json.dumps({"party": prediction})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
