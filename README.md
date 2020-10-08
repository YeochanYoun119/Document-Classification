# Document-Classification
Using supervised learning, create document classification program.
There are two types training documnets about AI. Some documnets are written in 2016 and the other are written in 2020.
Program reads a collection of test documents. 
Analyzing probability of vocabularies in the documnet and classfy them into proper year when they are written or topic is more related to other documents written in the same year.

# Training
## Create bag of words
Create bag of word for each training documents and counts the number of words appear in the document.
```
>>> vocab = create_vocabulary('./EasyFiles/', 1)
>>> create_bow(vocab, './EasyFiles/2016/1.txt')
=> {'a': 2, 'dog': 1, 'chases': 1, 'cat': 1, '.': 1}
>>> create_bow(vocab, './EasyFiles/2020/2.txt')
=> {'it': 1, 'is': 1, 'february': 1, '19': 1, ',': 1, '2020': 1, '.': 1}
```

## Load training data
Using bag of words created in previous step, load training documents and label them into proper years.
```
>>> vocab = create_vocabulary('./EasyFiles/', 1)
>>> load_training_data(vocab,'./EasyFiles/')
=> [{'label': '2016', 'bow':{'a': 2, 'dog': 1, 'chases': 1, 'cat': 1, '.': 1}},
    {'label': '2016', 'bow':{'hello': 1, 'world': 1}},
    {'label': '2020', 'bow':{'it': 1, 'is': 1, 'february': 1, '19': 1, ',': 1, '2020': 1, '.': 1}}]
```

## Prior Log Probability
This method should return the log probability LaTeX: \log{\hat{P}(y)}log ⁡ P ^ ( y ) of the labels in the training set In order to calculate these, you will need to count the number of documents with each label in the training data, found in the training/ subdirectory.
Because we usually have enough training documents in each class, we do not use add-1 smoothing here; instead we use the maximum likelihood estimate (MLE).   Note that the return values are the natural log of the probability. In a Naive Bayes implementation, we must contend with the possibility of underflow: this can occur when we take the product of very small floating point values. As such, all our probabilities in this program will be log probabilities, to avoid this issue.
```
>>> vocab = create_vocabulary('./corpus/training/', 2)
>>> training_data = load_training_data(vocab,'./corpus/training/')
>>> prior(training_data, ['2020', '2016'])
=> {'2020': -0.31939049933692143, '2016': -1.2967892172518587}
```

## Log Probability of a word, Given a label
This function returns a list consisting of the log conditional probability of all word types in a vocabulary (plus OOV) given a particular class label, log LaTeX: P\left(word\mid label\right)P ( w o r d ∣ l a b e l ).   To compute this probability, you must use add-1 smoothing, rather than the MLE (this is different from the prior) to avoid zero probability.
```
>>> vocab = create_vocabulary('./EasyFiles/', 1)
>>> training_data = load_training_data(vocab, './EasyFiles/')
>>> p_word_given_label(vocab, training_data, '2020')
=> {'a': -3.04, 'dog': -3.04, 'chases': -3.04, 'cat': -3.04, '.': -2.35, 
    'hello': -3.04, 'world': -3.04, 'it': -2.35, 'is': -2.35, 'february': -2.35, 
    '19': -2.35, ',': -2.35, '2020': -2.35, None: -3.04}
>>> p_word_given_label(vocab, training_data, '2016')
=> {'a': -1.99, 'dog': -2.4, 'chases': -2.4, 'cat': -2.4, '.': -2.4, 
    'hello': -2.4, 'world': -2.4, 'it': -3.09, 'is': -3.09, 'february': -3.09, 
    '19': -3.09, ',': -3.09, '2020': -3.09, None: -3.09}
```
## Train
```
>>> train('./EasyFiles/', 2)
=> { 'vocabulary': ['.', 'a'], 
     'log prior': {'2016': -0.41, '2020': -1.10}, 
     'log p(w|y=2016)': {'a': -1.30, '.': -1.70, None: -0.61}, 
     'log p(w|y=2020)': {'a': -2.30, '.': -1.61, None: -0.36} }
```


# Classify
Using this equation to find proper x and classify the documents.<br>
<img src="https://bit.ly/3lokLU5" align="center" border="0" alt="LaTeX: f(x) := argmax_{y\in Y} \hat{P}(x|y)\hat{P}(y)" width="311" height="25" />
```
>>> model = train('./corpus/training/', 2)
>>> classify(model, './corpus/test/2016/0.txt')
=> {'log p(y=2020|x)': -3906.35, 'log p(y=2016|x)': -3916.46, 'predicted y': '2020'}
```
