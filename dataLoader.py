from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import gzip
import numpy as np
import re
import pickle
from gensim.parsing.preprocessing import remove_stopwords
import matplotlib.pyplot as plt
import pickle


class AmznDataLoader:
        def __init__(self,path,maxSeqLength):
            self.maxSeqLength = maxSeqLength
            if ".json.gz" in path:
                df = self.getDF(path)
    #            df = self.getDF('./data/reviews_Musical_Instruments_5.json.gz')
                df = df[['reviewText', 'overall']]
                df['reviewText'] = df['reviewText'].apply(lambda x : self.title_parsing(x))
                X = df['reviewText']
                y = df['overall']-1
                self.weight_matrix = self.get_weight_matrix(X)
                X = self.indicesMatrix(X)
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                df1 = pd.read_fwf("./data/ss2Train.txt",names = ["overall","reviewText"], index_col = False)
                df2 = pd.read_fwf("./data/ss2Test.txt",names = ["overall","reviewText"], index_col = False)
                df['reviewText'] = df['reviewText'].apply(lambda x : self.title_parsing(x))
                X_train = df['reviewText']
                self.y_train = df['overall']
                self.weight_matrix = self.get_weight_matrix(X_train)
                print(type(weight_matrix))
                self.X_train = self.indicesMatrix(X_train)  
                
            np.save("weightmatrix.npy",self.weight_matrix)
        
        def parse(self, path):
            '''
            read files from .json.gz
            '''
            g = gzip.open(path, 'rb')
            for l in g:
                yield eval(l)
            
        def getDF(self, path):
            '''
            read files from .json.gz
            '''
            i = 0
            df = {}
            for d in self.parse(path):
                df[i] = d
                i += 1
            return pd.DataFrame.from_dict(df, orient='index')  
        
                
        def title_parsing(self, sentence):  
            '''    
            Tokenize input text to list of words after remove stopwords and words with only one letter
            
            Args:
            sentence (str): text
            
            Returns:
            list: a list of words in text
            ''' 
            sentence = re.sub('[^a-zA-Z]', ' ', str(sentence)).lower()
            tokens = remove_stopwords(sentence).split()  ## remove stop words, corpus size 52680            
            tokens = [word for word in tokens if len(word) >1 ]
            return tokens
        
        
      

        def buildCorpus(self, X):
            '''
            return a dictionary with word in X and its index in corpus as key and value respectively
            
            Args:
            X (Dataframe): dataframe with text
            
            Returns:
            Dict: key: word; value: index, based on its appearance order in X, starts from 0
            '''
            word2idx = {}
            idx2word = []
            for row in X:
                for word in row:
                    if word not in word2idx:
                        idx2word.append(word)                
                        word2idx[word] = len(idx2word) - 1
            return word2idx   
        
    
        def indicesMatrix(self, X):
            '''
            return a matrix (num_reviews, maxNumberWords) such that words transformed to its index in corpus dictionary
            
            Args:
            X (Dataframe): dataframe with text
            
            Returns:
            2darray           
            '''
            word2idx = self.buildCorpus(X)
            corpusSize = len(word2idx) 
        
            ### plot the histogram
            k = sorted(len(x) for x in X)
            plt.hist(k)
            ###            
#             maxNumberWords = sorted(len(x) for x in X)[-1]
#             print ("maximum", maxNumberWords)

            index_matrix = np.zeros((X.shape[0], self.maxSeqLength))          
            for i, row in enumerate(X):
                for j, word in enumerate(row):
#                     try:
#                         index_matrix[i,j] = word2idx[word]
#                         words_found += 1
#                     except KeyError:
#                         index_matrix[i,j] = corpusSize     

                    index_matrix[i,j] = word2idx[word]
                    if j >= self.maxSeqLength-1 : 
                        break
#            if maxNumberWords % 2 == 1:
#                x0 = np.full((index_matrix.shape[0], 1), maxNumberWords)
#                index_matrix = np.hstack((index_matrix, x0))
            return index_matrix
        
        def get_weight_matrix(self, X): #max norm for linear layer
            '''
            return (glove)embedding matrix (num_vocab, n_embedding) such that the 1st dimension corresponds to its index-1 in corpus dictionary
            Note that the word cannot be found in the pretrained glove returns random array as embedding
            
            Args:
            X (Dataframe): dataframe with text
            
            Returns:
            2darray   
            '''
            
            # load words and its embedding to a dictionary
            glove = {}
            embed_size = 50
            with open(f'./data/glove.6B.50d.txt', 'rb') as f:#
                for l in f:
                    line = l.decode().split()           
                    word = line[0]
                    vect = np.array(line[1:]).astype(np.float)
                    glove.update({word:vect})
            # generate weight matrix
            target_vocab = self.buildCorpus(X)
            matrix_len = len(target_vocab)
            weights_matrix = np.zeros((matrix_len, embed_size))#
            words_found = 0
            words_not_found = 0
            for i, word in enumerate(target_vocab):
                try: 
                    weights_matrix[i] = glove[word]
                    words_found += 1
                except KeyError:
                    words_not_found += 1
                    weights_matrix[i] = np.random.normal(scale=0.6, size=(embed_size,))
            print(words_not_found)
            return  weights_matrix