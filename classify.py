'''
Title:           Document Classification
Files:           classify.py
Course:          CS540, Spring 2020

Author:          Yeochan Youn
Email:           yyoun5@wisc.edu
'''

import glob
import math
import os

'''
loads the training data, estimates the prior distribution P(label) and class conditional distributions P(word|label) 
    - training_directory: dictionary form of data
    - cutoff: cutoff word count
 :return trained model
'''
def train(training_directory, cutoff):
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)

    ls = {'vocabulary': vocab, 'log prior': prior(training_data, ['2020', '2016']),
          'log p(w|y=2016)': p_word_given_label(vocab, training_data, '2016'),
          'log p(w|y=2020)': p_word_given_label(vocab, training_data, '2020')}

    return ls


'''
reate and return a vocabulary as a list of word types with counts >= cutoff in the training directory
    - training_directory: dictionary form of data
    - cutoff: cutoff word count
 :return list of vocabularies
'''
def create_vocabulary(training_directory, cutoff):
    f2016 = glob.glob(training_directory + '\\*\\*.txt')  # check all the txt files under filepath
    voc = {}
    for i in f2016:
        with open(i, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip() not in voc:
                    voc[line.strip()] = 1
                else:
                    voc[line.strip()] += 1
    pr = []
    for j in voc:
        if voc.get(j) >= cutoff:
            pr.append(j)
    return sorted(pr)


'''
create and return a bag of words Python dictionary from a single document
    - vocab: list of vocabularies
    - filepath: filepath where files are located
 :return list of vocabulary
'''
def create_bow(vocab, filepath):
    ls = {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            if line.strip() in vocab:
                if line.strip() in ls:
                    ls[line.strip()] += 1
                else:
                    ls[line.strip()] = 1
            else:
                if None in ls:
                    ls[None] += 1
                else:
                    ls[None] = 1
    return ls


'''
create and return training set (bag of words Python dictionary + label) from the files in a training directory
    - vocab: list of vocabularies
    - directory: directory where files are located
 :return list of dictionary
'''
def load_training_data(vocab, directory):
    ls = []
    for f in os.listdir(directory):
        for i in (os.listdir(os.path.join(directory, f))):
            ls.append({'label': f, 'bow': create_bow(vocab, os.path.join(directory, f, i))})  # add dictionary to list to return
    return ls


'''
given a training set, estimate and return the prior probability p(label) of each label 
    - training_data: list of dictionary with label and bow
    - label_list: list of labels
 :return dictionary of label paired with MLE
'''
def prior(training_data, label_list):
    ls = {};
    for lab in label_list:
        count = 0
        for i in range(len(training_data)):
            if training_data[i].get('label') == lab:
                count += 1
        ls[lab] = math.log(count / len(training_data))
    return ls

'''
given a training set and a vocabulary, estimate and return the class conditional distribution P ( word ∣ label ) over all words for the given label using smoothing
    - vocab: list of vocabularies
    - training_data: list of dictionary with label and bow
    - label_list: list of labels
 :return dictionary of words paired with calculated value
'''
def p_word_given_label(vocab, training_data, label):
    pg = {}
    for i in vocab:
        pg[i] = 1  # initialize all the values with 1 instead of add 1 later
    pg[None] = 1

    bt = [bow['bow'] for bow in training_data if bow['label'] == label]  # check for matching label
    for bow in bt:
        for w in bow:
            for k in range(bow[w]):
                pg[w] += 1

    t = 0
    for word in pg:
        t = t + pg[word]
    for word in pg:
        pg[word] = math.log(pg[word] / t)
    return pg


'''
given a trained model, predict the label for the test document
    - model: trained model
    - filepath: filepath where the files are located
 :return dictionary form of result
'''
def classify(model, filepath):
    vocab = model['vocabulary']
    fb = create_bow(vocab, filepath)
    p2016 = calc(fb, model['log p(w|y=2016)'], model['log prior']['2016'])  # calculate log probability of label
    p2020 = calc(fb, model['log p(w|y=2020)'], model['log prior']['2020'])
    ls = {}

    if p2016 > p2020:
        ls['predicted y'] = '2016'
    else:
        ls['predicted y'] = '2020'

    ls['log p(y=2016|x)'] = p2016
    ls['log p(y=2020|x)'] = p2020
    return ls


'''
calculate log probabilities of label
    - bow: dictionary of words paired with count
    - given: model['log p(w|y=2016)']
    - num: calculated prior number
 :return log probability
'''
def calc(bow, given, num):
    p = 0
    dic = {}
    for i in bow:
        dic[i] = bow[i]
    for w in dic:
        for i in range(dic[w]):
            p += given[w]
    p += num
    return p

print(train('./EasyFiles/', 2))
