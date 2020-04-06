from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer




class TfIdf():
    def  __init__(self, sentences):

        self.counter = CountVectorizer()
        self.count_vector = self.counter.fit_transform(sentences)

        self.tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
        self.tfidf.fit_transform(self.count_vector)


    def get_score(self, word: str) -> float:
        index = self.counter.get_feature_names().index(word)
        return self.tfidf.idf_[index]

if __name__ == "__main__":
    sentences = [
        "je suis une chèvre", "le jambon c'est bon",
        "le fromage de chèvre c'est bon", 'I de like dog', 'I love cat',
        'I interested in cat', "j'aime de vian"
    ]
    A = TfIdf(sentences)
    for word in A.counter.get_feature_names():
        s = A.get_score(word)
        print(word, s)
