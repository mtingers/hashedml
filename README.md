# HashedML
A machine learning library that uses a different approach: string hashing
(think hash tables) for classifying sequences.

# Installation

PyPI:
```
pip install -U hashedml
```

setup.py:
```
python setup.py build
python setup.py install
```

# Classification
HashedML takes the simple `fit(X, y)` / `predict(X)` approach.

Example:

```python
from hashedml import HashedML

model = HashedML()
iris_data = open('test-data/iris.data').read().split('\n')
for i in iris_data:
    i = i.split(',')
    X = i[:-1]
    y = i[-1]
    model.fit(X, y)

iris_test = open('test-data/iris.test').read().split('\n')
for i in irist_test:
    i = i.split(',')
    X = i[:-1]
    y = i[-1]
    # use test() to get accuracy
    prediction = model.test(X, y)
    # -or: normally you don't have 'y'
    prediction = model.predict(X)

print('accuracy: {}%'.format(model.accuracy()*100))

```

# Generative
HashedML can also generate data after learning.

Example:

```python
from collections import deque
from hashedml import HashedML

model = HashedML(nback=4)
token_q = deque(maxlen=model.nback)
tokens = TextBlob(open('training.text').read()).tokens

# Learn
for i in tokens:
    token_q.append(i)
    if len(token_q) != model.nback:
        continue
    X = list(token_q)[:-1]
    y = list(token_q)[-1]
    model.fit(X, y)

# Generate
output = model.generate(
    ('What', 'is'),
    nwords=500,
    separator=' '
)
print(output)
```

Example using `hashedml` test CLI program:
```bash
(venv) foo % hashedml generate 120 'Computer science' test-data/computerprogramming.txt
```
```
input-file: test-data/computerprogramming.txt
output:
Computer science abstracting the code, making it targetable to varying machine
instruction sets via compilation declarations and heuristics. The first
compiler for a programming language was developed by seven programmers,
including Adele Goldberg, in the 1970s. One of the first object-oriented
programming languages, Smalltalk, was developed by seven programmers,
including Adele in the 1970s. One of the first programming languages,
Smalltalk, was developed by Grace Hopper. When Hopper went to work on UNIVAC in
1949, she brought the idea of using compilers with her. Compilers harness the
power of computers to make programming easier by allowing programmers to
specify calculations by entering a formula using infix notation
( e.g., Y = X * 2 + 5 * X + 9
```
```bash
(venv) foo % hashedml generate 120 'Computer science' test-data/computerprogramming.txt
```
```
input-file: test-data/computerprogramming.txt
output:
Computer science abstracting the code, making it targetable to varying machine
instruction sets via compilation declarations and heuristics. The first
compiler for a programming language was developed by Grace Hopper. When Hopper
went to work on UNIVAC in 1949, she brought the idea of using compilers with
her. Compilers harness the power of computers to make programming easier by
allowing programmers to specify calculations by entering a formula using infix
notation ( e.g., Y = X * 2 + 5 * X + 9 ) for example. FORTRAN, the first widely
used high-level language to have a functional implementation which permitted
the abstraction of reusable blocks of code, came out in 1957 and many other
languages were soon developed that let the
```

# Variable X Input & Non-numerical X or Y
The X value can be of varying length/dimensions. For example, this is valid:
```python
X = (
    (1, 2, 3),
    (1, 2),
    (1, 2, 3, 4),
)
# y can be of different data types
y  = (
    'y1',
    2.0,
    'foostring'
)
```

All data is converted to strings. This is conterintuitive and different than
most machine learning libraries, but helps with working with variable X/y data.

# Examples

NOTE: Accuracy is impacted by the typical test/train/split scenario.
```bash
% for i in test-data/*.test; do echo -en "$i: "; data_file=$(echo $i|sed 's/.test/.data/g'); hashedml classify $data_file $i ; done

test-data/abalone.test: accuracy: 34.22%
test-data/allhypo.test: accuracy: 88.42%
test-data/anneal.test: accuracy: 78.87%
test-data/arrhythmia.test: accuracy: 38.00%
test-data/breast-cancer.test: accuracy: 64.21%
test-data/bupa.test: accuracy: 55.26%
test-data/glass.test: accuracy: 33.80%
test-data/iris.test: accuracy: 95.83%
test-data/parkinsons_updrs.test: accuracy: 83.71%
test-data/soybean-large.test: accuracy: 44.12%
test-data/tic-tac-toe.test: accuracy: 82.13%
```

# Method Parameter Notes

* `HashedML.predict(X, return_one=True)` -- Return a single highest rated item
* `HashedML.predict(X, return_one=False)` -- Return a list of top 10 predictions
* `HashedML(nback=4)` -- Used with `generate()` logic for tracking history of
    generated items and what to feed next as X input.
* `HashedML.generate(X, nwords=100)` -- Run generation 100 times
* `HashedML.generate(X, stm=True)` -- Use short-term memory logic to try to keep
    on topic.
* `HashedML.generate(X, separator=' ')` -- Inspect generated items and make sure
    it ends with this separator. An example would be if input text data
    stripped out spaces (e.g. output could be `Hello,world.Nospaces` or with
    separator specified: `Hello, world. No spaces`)

# Scikit-learn Comparison

Here is a test of various classifiers from scikit-learn, trained/tested against
the same datasets from the examples section:

```
test-data/abalone.test:
                  KNeighborsClassifier(n_neighbors=3) 50.72%
                        SVC(C=0.025, kernel='linear') 51.22%
                                    SVC(C=1, gamma=2) 51.59%
                  DecisionTreeClassifier(max_depth=5) 54.76%
   RandomForestClassifier(max_depth=5, max_features=1 53.24%
               MLPClassifier(alpha=1, max_iter=10000) 52.31%
                                         GaussianNB() 51.44%
                                              AVERAGE 52.18%
test-data/allhypo.test:
                  KNeighborsClassifier(n_neighbors=3) 93.68%
                        SVC(C=0.025, kernel='linear') 93.46%
                                    SVC(C=1, gamma=2) 91.53%
                  DecisionTreeClassifier(max_depth=5) 98.61%
   RandomForestClassifier(max_depth=5, max_features=1 91.85%
               MLPClassifier(alpha=1, max_iter=10000) 96.78%
                                         GaussianNB() 20.47%
                                              AVERAGE 83.77%
test-data/anneal.test:
                  KNeighborsClassifier(n_neighbors=3) 81.51%
                        SVC(C=0.025, kernel='linear') 97.36%
                                    SVC(C=1, gamma=2) 73.21%
                  DecisionTreeClassifier(max_depth=5) 97.74%
   RandomForestClassifier(max_depth=5, max_features=1 78.11%
               MLPClassifier(alpha=1, max_iter=10000) 93.21%
                                         GaussianNB() 67.17%
                                              AVERAGE 84.04%
test-data/arrhythmia.test:
                  KNeighborsClassifier(n_neighbors=3) 57.33%
                        SVC(C=0.025, kernel='linear') 66.67%
                                    SVC(C=1, gamma=2) 51.33%
                  DecisionTreeClassifier(max_depth=5) 51.33%
   RandomForestClassifier(max_depth=5, max_features=1 54.00%
               MLPClassifier(alpha=1, max_iter=10000) 58.67%
                                         GaussianNB() 17.33%
                                              AVERAGE 50.95%
test-data/breast-cancer.test:
                  KNeighborsClassifier(n_neighbors=3) 68.42%
                        SVC(C=0.025, kernel='linear') 71.58%
                                    SVC(C=1, gamma=2) 73.68%
                  DecisionTreeClassifier(max_depth=5) 64.21%
   RandomForestClassifier(max_depth=5, max_features=1 73.68%
               MLPClassifier(alpha=1, max_iter=10000) 65.26%
                                         GaussianNB() 75.79%
                                              AVERAGE 70.38%
test-data/bupa.test:
                  KNeighborsClassifier(n_neighbors=3) 63.16%
                        SVC(C=0.025, kernel='linear') 64.91%
                                    SVC(C=1, gamma=2) 52.63%
                  DecisionTreeClassifier(max_depth=5) 61.40%
   RandomForestClassifier(max_depth=5, max_features=1 59.65%
               MLPClassifier(alpha=1, max_iter=10000) 66.67%
                                         GaussianNB() 59.65%
                                              AVERAGE 61.15%
test-data/glass.test:
                  KNeighborsClassifier(n_neighbors=3) 61.97%
                        SVC(C=0.025, kernel='linear') 36.62%
                                    SVC(C=1, gamma=2) 54.93%
                  DecisionTreeClassifier(max_depth=5) 66.20%
   RandomForestClassifier(max_depth=5, max_features=1 64.79%
               MLPClassifier(alpha=1, max_iter=10000) 25.35%
                                         GaussianNB() 57.75%
                                              AVERAGE 52.52%
test-data/iris.test:
                  KNeighborsClassifier(n_neighbors=3) 95.83%
                        SVC(C=0.025, kernel='linear') 95.83%
                                    SVC(C=1, gamma=2) 93.75%
                  DecisionTreeClassifier(max_depth=5) 95.83%
   RandomForestClassifier(max_depth=5, max_features=1 95.83%
               MLPClassifier(alpha=1, max_iter=10000) 95.83%
                                         GaussianNB() 93.75%
                                              AVERAGE 95.24%
test-data/parkinsons_updrs.test:
                  KNeighborsClassifier(n_neighbors=3) 89.27%
                        SVC(C=0.025, kernel='linear') 98.62%
                                    SVC(C=1, gamma=2) 62.21%
                  DecisionTreeClassifier(max_depth=5) 27.63%
   RandomForestClassifier(max_depth=5, max_features=1 43.26%
               MLPClassifier(alpha=1, max_iter=10000) 93.11%
                                         GaussianNB() 95.35%
                                              AVERAGE 72.78%
test-data/soybean-large.test:
                  KNeighborsClassifier(n_neighbors=3) 82.35%
                        SVC(C=0.025, kernel='linear') 71.57%
                                    SVC(C=1, gamma=2) 18.63%
                  DecisionTreeClassifier(max_depth=5) 73.53%
   RandomForestClassifier(max_depth=5, max_features=1 75.49%
               MLPClassifier(alpha=1, max_iter=10000) 62.75%
                                         GaussianNB() 82.35%
                                              AVERAGE 66.67%
test-data/tic-tac-toe.test:
                  KNeighborsClassifier(n_neighbors=3) 83.07%
                        SVC(C=0.025, kernel='linear') 70.53%
                                    SVC(C=1, gamma=2) 70.53%
                  DecisionTreeClassifier(max_depth=5) 90.91%
   RandomForestClassifier(max_depth=5, max_features=1 76.49%
               MLPClassifier(alpha=1, max_iter=10000) 84.33%
                                         GaussianNB() 73.04%
                                              AVERAGE 78.41%
```
