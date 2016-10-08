import collections  # optional, but we found the collections.Counter object useful
import scipy.sparse as sp
import numpy as np
import re
import random

def load_federalist_corpus(filename):
    """ Load the federalist papers as a tokenized list of strings, one for each eassay"""
    with open(filename, "rt") as f:
        data = f.read()
    papers = data.split("FEDERALIST")

    # all start with "To the people of the State of New York:" (sometimes . instead of :)
    # all end with PUBLIUS (or no end at all)
    locations = [(i, [-1] + [m.end() + 1 for m in re.finditer(r"of the State of New York", p)],
                  [-1] + [m.start() for m in re.finditer(r"PUBLIUS", p)]) for i, p in enumerate(papers)]
    papers_content = [papers[i][max(loc[1]):max(loc[2])] for i, loc in enumerate(locations)]

    # discard entries that are not actually a paper
    papers_content = [p for p in papers_content if len(p) > 0]

    # replace all whitespace with a single space
    papers_content = [re.sub(r"\s+", " ", p).lower() for p in papers_content]

    # add spaces before all punctuation, so they are separate tokens
    punctuation = set(re.findall(r"[^\w\s]+", " ".join(papers_content))) - {"-", "'"}
    for c in punctuation:
        papers_content = [p.replace(c, " " + c + " ") for p in papers_content]
    papers_content = [re.sub(r"\s+", " ", p).lower().strip() for p in papers_content]

    authors = [tuple(re.findall("MADISON|JAY|HAMILTON", a)) for a in papers]
    authors = [a for a in authors if len(a) > 0]

    numbers = [re.search(r"No\. \d+", p).group(0) for p in papers if re.search(r"No\. \d+", p)]

    return papers_content, authors, numbers

def tfidf(docs):
    """
    Create TFIDF matrix.  This function creates a TFIDF matrix from the
    docs input.

    Args:
        docs: list of strings, where each string represents a space-separated
              document

    Returns: tuple: (tfidf, all_words)
        tfidf: sparse matrix (in any scipy sparse format) of size (# docs) x
               (# total unique words), where i,j entry is TFIDF score for
               document i and term j
        all_words: list of strings, where the ith element indicates the word
                   that corresponds to the ith column in the TFIDF matrix
    """
    vocab = {}
    df = {}
    regex = re.compile("\s+")
    count = 0
    for doc in docs:
        terms = re.split(regex, doc)
        for term in set(terms):
            if len(term) > 0:
                if term not in vocab:
                    vocab[term] = count  # (index, df)
                    df[term] = 1
                    count += 1
                else:
                    df[term] += 1
    num_docs = len(docs)
    scores = []
    for i in range(0, num_docs):
        scores.append({})

    for index in range(0, num_docs):
        terms = re.split(regex, docs[index])
        for term, tf in collections.Counter(terms).most_common():
            if len(term) > 0:
                term_index = vocab[term]
                score = float(tf) * np.log(float(num_docs) / float(df[term]))
                if score > 0.0:
                    scores[index][term_index] = score

    i_list = []
    j_list = []
    data = []

    for i in range(0, num_docs):
        for j, score in scores[i].iteritems():
            i_list.append(i)
            j_list.append(j)
            data.append(score)

    matrix = sp.csr_matrix((data, (i_list, j_list)), shape=(num_docs, len(vocab)))
    reverse_map = {v: k for k, v in vocab.iteritems()}
    return matrix, reverse_map.values()


def cosine_similarity(X):
    """
    Return a matrix of cosine similarities.

    Args:
        X: sparse matrix of TFIDF scores or term frequencies

    Returns:
        M: dense numpy array of all pairwise cosine similarities.  That is, the
           entry M[i,j], should correspond to the cosine similarity between the
           ith and jth rows of X.
    """
    matrix = X.dot(X.transpose()).todense()
    mat_len = len(matrix)
    norms = [0] * mat_len
    for i in range(0, mat_len):
        norms[i] = 1.0 / np.sqrt(matrix.item((i, i)))
    norm_mat = np.matrix(norms)
    return np.multiply(norm_mat.transpose().dot(norm_mat), matrix)

class LanguageModel:
    def __init__(self, docs, n):
        """
        Initialize an n-gram language model.

        Args:
            docs: list of strings, where each string represents a space-separated
                  document
            n: integer, degree of n-gram model
        """
        self.n = n
        self.dict = {}
        self.vocab = set()
        self.sum_index = "*sum*"
        regex = re.compile("\s+")
        count = 0
        for doc in docs:
            terms = re.split(regex, doc)
            for term in terms:
                if term not in self.vocab:
                    self.vocab.add(term)
            for i in range(0, len(terms) - n + 1):
                end = i+n-1
                t = tuple(terms[i:end])
                if t not in self.dict:
                    self.dict[t] = {}
                    self.dict[t][self.sum_index] = 0
                self.dict[t][self.sum_index] += 1
                end_term = terms[end]
                if end_term not in self.dict[t]:
                    self.dict[t][end_term] = 1
                else:
                    self.dict[t][end_term] += 1
        self.D = len(self.vocab)

    def perplexity(self, text, alpha=1e-3):
        """
        Evaluate perplexity of model on some text.

        Args:
            text: string containing space-separated words, on which to compute
            alpha: constant to use in Laplace smoothing

        Note: for the purposes of smoothing, the dictionary size (i.e, the D term)
        should be equal to the total number of unique words used to build the model
        _and_ in the input text to this function.

        Returns: perplexity
            perplexity: floating point value, perplexity of the text as evaluted
                        under the model.
        """
        regex = re.compile("\s+")
        terms = re.split(regex, text)
        n = self.n
        D = self.D
        logp_sum = 0.0

        for term in terms:
            if term not in self.vocab:
                D += 1

        for i in range(0, len(terms) - n + 1):
            end = i + n - 1
            t = tuple(terms[i:end])
            end_term = terms[end]
            c = 0.0
            c_sum = 0.0
            if t in self.dict and end_term in self.dict[t]:
                c = self.dict[t][end_term]
                c_sum = self.dict[t][self.sum_index]
            p = float(c + alpha) / float(c_sum + alpha * D)
            log_p = np.log2(p)
            logp_sum += log_p
        return np.power(2, -logp_sum/len(terms))

    def gen_beginning(self):
        t_sum = len(self.dict)
        rand = random.randint(0, t_sum)
        i = 0
        for t in self.dict.iterkeys():
            if i == rand:
                return list(t)
            i += 1

    def sample(self, k):
        """
        Generate a random sample of k words.

        Args:
            k: integer, indicating the number of words to sample

        Returns: text
            text: string of words generated from the model.
        """
        result = ""
        current = self.gen_beginning()
        for i in range(0, k):
            result += current[0] + " "
            t = tuple(current)
            if t in self.dict:
                c_sum = self.dict[t][self.sum_index]
                rand = random.randint(0, c_sum)
                new_term = ""
                for term, count in self.dict.iteritems():
                    if rand > count:
                        rand -= count
                    else:
                        new_term = term
                        break
                current.remove(current[0])
                current.append(new_term)
            else:
                current = self.gen_beginning()
        return result

# AUTOLAB_IGNORE_START
papers, authors, numbers = load_federalist_corpus("pg18.txt")
# AUTOLAB_IGNORE_STOP

### AUTOLAB_IGNORE_START
data = [
    "the goal of this lecture is to explain the basics of free text processing",
    "the bag of words model is one such approach",
    "text processing via bag of words"
]

X_tfidf, words = tfidf(data)
print X_tfidf.todense()
print words

### AUTOLAB_IGNORE_STOP

### AUTOLAB_IGNORE_START
print cosine_similarity(X_tfidf)
### AUTOLAB_IGNORE_STOP

### AUTOLAB_IGNORE_START
hamilton = []
for p, a in zip(papers, authors):
    if a[0] == 'HAMILTON':
        hamilton.append(p)

l_hamilton = LanguageModel(hamilton, 3)
print l_hamilton.perplexity(papers[0])
print l_hamilton.sample(100)
### AUTOLAB_IGNORE_STOP