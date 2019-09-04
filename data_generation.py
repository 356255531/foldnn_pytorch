import _pickle as pickle
import re
import numpy as np
import gensim
import pdb

TRAIN_DATA_FILE = 'data/stsa.binary.phrases.train'
DEV_DATA_FILE = 'data/stsa.binary.dev'
TEST_DATA_FILE = 'data/stsa.binary.test'
W2V_FILE = 'weights/GoogleNews-vectors-negative300.bin'

GOOGLE_VOCAB_MAT_PATH = 'data/google_vocab_mat.pkl'
TRAIN_PATH = 'data/train.pkl'
DEV_PATH = 'data/dev.pkl'
TEST_PATH = 'data/test.pkl'
LOGIC_TRAIN_PATH = 'data/logic_train.pkl'
LOGIC_DEV_PATH = 'data/logic_dev.pkl'
LOGIC_TEST_PATH = 'data/logic_test.pkl'


def regularize_string(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string


def read_corpus(file_path):
    """
    Read in the corpus
    :param file_path:
    :return:
        corpus: List of strings
        labels: List of 0 and 1
        fre_vocab: Word frequency vocabulary
    """
    max_sentence_length = 0
    corpus, labels, fre_vocab = [], [], {}
    with open(file_path, 'rb') as f:
        lines = f.readlines()
        for l in lines:
            line = l.decode('utf-8').strip().lower()
            x = regularize_string(line[2:])
            y = line[0]
            if len(x):
                labels.append(int(y))
                corpus.append(x)

                words = x.split()
                num_words = len(words)
                max_sentence_length = num_words if num_words > max_sentence_length else max_sentence_length

                for word in words:
                    if word in fre_vocab.keys():
                        fre_vocab[word] += 1
                    else:
                        fre_vocab[word] = 1
    return corpus, labels, fre_vocab, max_sentence_length


def load_bin_vec(file_name, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    model = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=True)
    word_vectors = {}
    for word in vocab:
        try:
            word_vectors[word] = model[word]
        except Exception:
            print('{} not in the word2vector!'.format(word))
    return word_vectors


def add_unknown_words_inp(word_vectors, fre_vocab, min_df=1, embedding_size=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in fre_vocab:
        if word not in word_vectors and fre_vocab[word] >= min_df:
            word_vectors[word] = np.random.uniform(-0.25, 0.25, embedding_size)


def create_word_vectors_mat(word_vectors, word_vocab, embedding_size=300):
    '''
    Transform the word in sorted word_vocab to matrix and append the existence of but at the last two rows
    :param word_vectors: dict, word(key), vector(value)
    :param word_vocab: sorted words in the corpus
    :param embedding_size: int
    :return: numpy.array
    '''
    word_vectors_mat = []
    word_vectors_mat.append(np.random.uniform(-0.25, 0.25, embedding_size))  # Appearance of but
    word_vectors_mat.append(np.random.uniform(-0.25, 0.25, embedding_size))  # No existence of but
    word_vectors_mat.append(np.random.uniform(-0.25, 0.25, embedding_size))  # The absence word
    word_vectors_mat.append(np.random.uniform(-0.25, 0.25, embedding_size))  # The placeholder of short sentences

    for word in word_vocab:
        word_vectors_mat.append(word_vectors[word])

    word_vectors_mat = np.array(word_vectors_mat)
    return word_vectors_mat


def index_corpus(corpus, labels, word_vocab, max_sentence_length):
    corpus_word_indices = []
    final_label = []
    for sentence, label in zip(corpus, labels):
        words = sentence.split()
        word_indices = []

        #  Add the logic symbol at the begin of word_indices
        if 'but' in words:
            word_indices.append(0)
        else:
            word_indices.append(1)

        for word in words:
            if word in word_vocab:
                word_indices.append(word_vocab.index(word) + 4)
            else:
                word_indices.append(2)
        if len(word_indices) <= max_sentence_length + 1:
            for i in range(max_sentence_length + 1 - len(word_indices)):
                word_indices.append(3)
        else:
            word_indices = word_indices[:(max_sentence_length + 1)]
        corpus_word_indices.append(word_indices)
        final_label.append(label)

    logic_corpus_word_index_mat = np.array(corpus_word_indices)
    corpus_word_index_mat = np.array(corpus_word_indices)[:, 1:]
    final_label_mat = np.array(final_label).reshape(-1, 1)

    return corpus_word_index_mat, logic_corpus_word_index_mat, final_label_mat


def separate_pos_neg(feature, label):
    pos_input, neg_input = [], []
    for idx, y in enumerate(label):
        if y == 0:
            neg_input.append(feature[idx])
        else:
            pos_input.append(feature[idx])
    return np.array(pos_input), np.array(neg_input)


def create_dataset():
    print("loading data...")
    train_corpus, train_label, fre_vocab, max_sentence_length = read_corpus(TRAIN_DATA_FILE)
    dev_corpus, dev_label, _, _ = read_corpus(DEV_DATA_FILE)
    test_corpus, test_label, _, _ = read_corpus(TEST_DATA_FILE)
    print('data loaded！')
    print("number of sentences: {}".format(len(train_corpus)))
    print("vocab size: {}".format(len(fre_vocab)))

    print("loading word2vec vectors...")
    google_word_vectors = load_bin_vec(W2V_FILE, fre_vocab)
    num_word = len(google_word_vectors)
    print("word2vec loaded!")
    print("num words already in word2vec: {}".format(num_word))

    print('Adding frequent but now existing words...')
    add_unknown_words_inp(google_word_vectors, fre_vocab, 10)
    print('{} random vectors added.'.format(len(fre_vocab) - num_word))

    # print('Adding random vectors...')
    # rand_vectors = {}
    # add_unknown_words(rand_vectors, fre_vocab)
    # random_word_vocab = sorted(rand_vectors.keys())
    # print('{} random vectors added!'.format(len(rand_vectors)))

    print('Creating vocabulary matrix...')
    google_word_vocab = sorted(google_word_vectors.keys())
    google_vocab_mat = create_word_vectors_mat(google_word_vectors, google_word_vocab)
    print('Vocabulary matrix created!')

    print('Indexing corpus...')
    google_train_word_index_mat, logic_google_train_word_index_mat, train_label = index_corpus(train_corpus, train_label, google_word_vocab, max_sentence_length)
    # random_train_input, random_vocab_mat = index_corpus(train_corpus, random_word_vocab, rand_vectors)
    google_dev_word_index_mat, logic_google_dev_word_index_mat, dev_label = index_corpus(dev_corpus, dev_label, google_word_vocab, max_sentence_length)
    google_test_word_index_mat, logic_google_test_word_index_mat, test_label = index_corpus(test_corpus, test_label, google_word_vocab, max_sentence_length)
    print('Corpus indexed!')

    pickle.dump(google_vocab_mat, open(GOOGLE_VOCAB_MAT_PATH, "wb"))
    pickle.dump([google_train_word_index_mat, train_label], open(TRAIN_PATH, "wb"))
    pickle.dump([google_dev_word_index_mat, dev_label], open(DEV_PATH, "wb"))
    pickle.dump([google_test_word_index_mat, test_label], open(TEST_PATH, "wb"))
    pickle.dump([logic_google_train_word_index_mat, train_label], open(LOGIC_TRAIN_PATH, "wb"))
    pickle.dump([logic_google_dev_word_index_mat, dev_label], open(LOGIC_DEV_PATH, "wb"))
    pickle.dump([logic_google_test_word_index_mat, test_label], open(LOGIC_TEST_PATH, "wb"))
    print("Dataset created!")


if __name__ == "__main__":
    create_dataset()
