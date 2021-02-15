# NLP_Tweets
We will be using trump tweets from a kaggle dataset to generate new trump tweets https://www.kaggle.com/ahmedterry/trump-tweets-eda-nlp-sentiments-analysis

for out first model we used spacy to tokenize the documents and we use spaces doc.vocab to convert the tokens to hashes and vice versa.
then we go through all the unique tweets and split up our data so that the first two tokens become the in_sequence and the next third token becomes our "labled data" out_sequence. then contiuneing to do that for the whole tweet putting the out_sequence into the in_sequence and then getting the next word to become the out_sequence.

after we create a sequce we append it to our 