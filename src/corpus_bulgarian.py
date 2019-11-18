from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["Иванчо получи нов компютър и Иванчо стана нов човек"]

# create the transform
vectorizer1 = CountVectorizer()
# tokenize and build vocab
vectorizer1.fit(text)

# summarize
print(vectorizer1.vocabulary_)
# encode document
vector1 = vectorizer1.transform(text)

# summarize encoded vector
print(vector1.shape)
print(vector1.toarray())


corpus = [
     'това е документ номер едно.',
     'а това е документ номер две.',
     'а това е документ три',
     'дали това е документ три', 
      ]

vectorizer2 = CountVectorizer()
vectorizer2.fit(corpus)
print(vectorizer2.vocabulary_)
vector2 = vectorizer2.transform(corpus)
print(vector2.shape)
print(vector2.toarray())


#{'това': 5, 'документ': 2, 'номер': 4, 'едно': 3, 'две': 1, 'три': 6, 'дали': 0}

#(4, 7)
#[[0 0 1 1 1 1 0]
# [0 1 1 0 1 1 0]
# [0 0 1 0 0 1 1]
# [1 0 1 0 0 1 1]]