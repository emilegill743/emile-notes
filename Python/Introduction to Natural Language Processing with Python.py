#####INTRODUCTION TO NATURAL LANGUAGE PROCESSING WITH PYTHON#####

'''Regular expressions and word tokenization'''

    '''Introduction to regular expressions'''

        # What is NLP?

            # Field of study focussed on making sense of language using statistics and computing
            # Applications: Chatbots, Translation, Sentiment Analysis etc.

        # Regular expressions

            # Strings with special pattern-matching syntax
            # Applications:
                # Find all web links in a document
                # Parse email addresses, remove/replace unwanted characters

            import re

            re.match('abc', 'abcdef')

            # Match any word
            word_regex = '\w+'
            re.match(word_regex, 'hi there!')

        # Common regex patterns

            '\w+'   # word
            '\d'    # digit
            '\s'    # space
            '.*'    # wildcard
            '+' or '*' # greedy match
            '\S'    # not space
            '[a-z]' # lowercase group

        # re Module

            re.split   # split string on regex
            re.findall # find all patterns in string
            re.search  # search for pattern 
            re.match   # match entire string or substring based on pattern

            # Syntax - pattern first, string second

            # May return iterator, string, or match object

    '''Introduction to tokenisation'''

        # Tokenisation - turning a string or document in tokens (smaller chunks)

            # Step in preparing text for NLP
            # Many theories and rules e.g. breaking out words or sentences, seperating punctuation, seperating hastags in a tweet

        # nltk - natural language toolkit library

            from nltk.tokensize import word_tokenize

            word_tokenize("Hi there!")

            # Out : ['Hi', 'there', '!']

        # Why tokenize?

            # Easier to map part of speech
            # Matching common words
            # Removing unwanted tokens
        
        # Other nltk tokenizers
            
            sent_tokenize   # tokenize a document into sentences
            regexp_tokenize # tokenize a string or document based on regex
            TweetTokenizer  # special class for tweet tokenization

    '''Advanced tokenization with NLTK and regex'''

        # OR represented by |
        # Define a group using ()
        # Define explicit character ranges using []

        import re

        match_digits_and_words = ('(\d+|\w+)')

        re.findall(match_digits_and_words, 'He has 11 cats.')
        # Out : ['He', 'has', '11', 'cats']

        # Regex ranges and groups

            '[A-Za-z]+'     # upper and lowercase English alphabet
            '[0-9]'         # numbers 0-9
            '[A-Za-z\-\.]+' # upper and lowercase English alphabet, '-' and '.'
            '(a-z)'         # a, '-' and z
            '(\s+|,)'       # spaces or comma

            # Escape characters required for - and . in range [] definitions, as these are reserved characters

            import re

            my_str = 'match lowercase spaces nums like 12, but no commas'

            re.match('[a-z0-9 ]+', my_str)

            # Out[3]: <_sre.SRE_Match object; 
            #   span=(0, 42), match='match lowercase spaces nums like 12'>

    '''Charting word length with NLTK'''

        from matplotlib import pyplot as plt
        from nltk.tokensize import word_tokensize

        words = word_tokensize("This is a pretty cool tool!")
        word_lengths = [len(w) for w in words]

        plt.hist(word_lengths)
        plt.show()

'''Simple topic identification'''

    '''Word counts with bag-of-words'''

        # Basic method for finding topics in a text
        # Tokenize text, count all tokens, fdetermine frequency of word usage

        from nltk.tokenize import word_tokenize
        from collections import Counter

        Counter(word_tokenize(
                """The cat is in the box. The cat likes the box. 
                 The box is over the cat."""))

        # Out[3]: 
        # Counter({'.': 3,
        #         'The': 3,
        #         'box': 3,
        #         'cat': 3,
        #         'in': 1,
        #         ...
        #         'the': 3})

        counter.most_common(2)
        # Out[4]: [('The', 3), ('box', 3)]

    '''Simple text preprocessing'''

        # Helps make fr better input data for machine learning, statistical models
        # Examples: 
            # tokenization to create bag of words
            # lowercasing words
            # lemmatization/stemming - shorten words to root stems
            # removing stop words, punctuation, unwanted tokens

        from nltk.corpus import stopwords

        text = """The cat is in the box. The cat likes the box.
                  The box is over the cat."""

        # Only return alphabetic strings
        tokens = [w for w in word_tokenize(text.lower())
                  if w.isalpha()]

        # Remove stop words
        no_stops = [t for t in tokens
                    if t not in stopwords.words('english')]

        # Count most common words
        Counter(no_stops).most_common(2)   

    '''Introduction to gensim'''

        # Popular open-source NLP library
        # Top academic models to perform complex tasks

            # Building document or word vectors
            # Performing topic identification and document comparison

        # Creating a gensim dictionary

            from gensim.corpora.dictionary import Dictionary
            from nltk.tokenize import word_tokenize

            # Corpus is a set of text used to help perform NLP tasks

            my_documents = ['The movie was about a spaceship and aliens.',
                             'I really liked the movie!',
                             'Awesome action scenes, but boring characters.',
                             'The movie was awful! I hate alien films.',
                             'Space is cool! I liked the movie.',
                             'More space films, please!',]
            
            tokenized_docs = [word_tokenize(doc.lower())
                              for doc in my_documents]

            dictionary = Dictionary(tokenized_docs)

            # Dictionary of all tokens and respective ids
            dictionary.token2id

            # Creating a gensim corpus -list of tuples (id, freq)
            corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

            # gensim models can be easily saved, updated and reused

    '''Tf-idf with gensim'''

        # Term frequency - inverse document frequency
        # Allows you to determine the most important words in each document

        # Each corpus may have shared words beyond just stopwords
        # These words should be down-weighted in importance
        # E.g. "Sky" for an astronomer will be frequent but unimportant
        # Ensures most common words don't show up as keywords
        # Keeps document specific frequent words weighted high

        from gensim.models.tfidfmodel import TfidfModel

        tfidf = TfidfModel(corpus)

        # Reference particular document (id, weight)
        tfidf[corpus[1]]

        # Out: [(0, 0.1746298276735174),
        #       (1, 0.1746298276735174),
        #       (9, 0.29853166221463673),
        #       (10, 0.7716931521027908),
        #       ...

    '''Named Entity Recognition'''

        # NER - NLP task to identify important named entities in text
        # e.g. people, places, organisations
        #      dates, states, works of art etc.

        # Who? What? When? Where?

        # nltk and Stanford CoreNLP Library
            # Integrated into Python via nltk
            # Java based
            # Support for NER as well as coreference and dependency trees

        import nltk

        sentence = '''In New York, I like to ride the Metro to visit MOMA 
                      and some restaurants rated well by Ruth Reichl.'''

        tokenized_sent = nltk.word_tokenize(sentence)
        tagged_sent = nltk.pos_tag(tokenized_sent)
        tagged_sent[:3]

        # Out[5]: [('In', 'IN'), ('New', 'NNP'), ('York', 'NNP')]
        # NNP - proper noun singular

        # Named Entity Chunk Tree
        ne_chunk(tagged_sent)

        '''Introduction to SpaCy'''

            # SpaCy - NLP library similar to genism, with different implementations
            
            # Focus on creating NLP pipelines to generate models and corpora
            # Open source, with extra libraries and tools, e.g. Displacy (visualisation)

            # SpaCy NER

                import spacy

                # Download pre-trained word vectors
                nlp = spacy.load('en')

                # Entity recogniser object
                nlp.entity

                doc = nlp("""Berlin is the capital of Germany; 
                            and the residence of Chancellor Angela Merkel.""")

                doc.ents
                # Out: (Berlin, Germany, Angela Merkel)

                print(doc.ents[0], doc.ents.label_)
                # Out: Berlin GPE
                # GPE - geopolitical entity

            # Why use SpaCy

                # Easy pipeline creation
                # Different entity types compared to nltk
                # Informal language corpora - easy to find entities in tweets and chat messages
                # Quickly growing

        '''Multilingual NER with polyglot'''

    '''Building a "fake news" classifier'''

        # Supervised learning -

            # Form of machine learning
            # Predefined training data
            # Data has label (or outcome) we want the model to learn

        # Supervised learning steps

            # Collect and preprocess our data
            # Determine a label (Example: Movie genre)
            # Split data into training and test sets
            # Extract features from the text to help predict the label
                # Bag of words vector built into scikit-learn
            # Evaluate trained model using test set

        '''Building word count vectors with scikit-learn'''

            # Count Vectorizer for text classification
                
                import pandas as pd
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import CountVectorizer
                
                df = ... # Load data into DataFrame
                y = df['Sci-Fi']
                
                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(
                                                df['plot'], y, 
                                                test_size=0.33, 
                                                random_state=53)

                # Convert text into bag-of-words vectors, remove stop words
                count_vectorizer = CountVectorizer(stop_words='english')

                # Generates mapping of words with ids, vectors representing word freq.
                count_train = count_vectorizer.fit_transform(X_train.values)

                # Operates differently for different models
                # Generally:
                #      - Fit finds parameters in data
                #      - Transform applies models underlying algorithm/approximation

                # Transform test data
                count_test = count_vectorizer.transform(X_test.values)

                # Print first 10 features of count_vectorizer
                print(count_vectorizer.get_feature_names()[:10])

            # Tfidf Vectorizer for text classification

                from sklearn.feature_extraction.text import TfidfVectorizer

                # Initialize a TfidfVectorizer object
                tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

                # Transform the training data 
                tfidf_train = tfidf_vectorizer.fit_transform(X_train.values)

                # Transform the test data 
                tfidf_test = tfidf_vectorizer.transform(X_test.values)

                # Print the first 10 features
                print(tfidf_vectorizer.get_feature_names()[:10])

                # Print the first 5 vectors of the tfidf training data
                print(tfidf_train.A[:5])

            # Inspecting the vectors

                # Create the CountVectorizer DataFrame: count_df
                count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

                # Create the TfidfVectorizer DataFrame: tfidf_df
                tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

                # Print the head of count_df
                print(count_df.head())

                # Print the head of tfidf_df
                print(tfidf_df.head())

                # Calculate the difference in columns: difference
                difference = set(count_df.columns) - set(tfidf_df.columns)
                print(difference)

                # Check whether the DataFrames are equal
                print(count_df.equals(tfidf_df))

        '''Training and testing a classification model with scikit-learn'''

            # Naive Bayes Model

                # Commonly used in testing NLP classification problems
                # Basis in probability
                # Given a particular piece of data, how likely is a particular outcome
                # Examples:
                    # If the plot has a spaceshi[, how likely is it to be sci-fy?
                    # Given a spaceship and an alien, how likely now is it sci-fy?

            from sklearn.naive_bayes import MultinomialNB
            from sklearn import metrics

            nb_classifier  = MultinomialNB()
            nb_classifier.fit(count_train, y_train)

            # Make prediction of test data based on trained model
            pred = nb_classifier.predict(count_test)

            # Evaluate accuracy of prediction
            metrics.accuracy_score(y_test, pred)

            # Top left, bottom right show true classifications
            metrics.confusion_matrix(y_test, pred, labels=[0,1])

        '''Simple NLP, Complex Problems'''

            




