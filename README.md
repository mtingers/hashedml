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

```bash
% for i in test-data/*.test; do echo -en "$i: "; data_file=$(echo $i|sed 's/.test/.data/g'); hashedml classify $data_file $i ; done

test-data/abalone.test: accuracy: 100.0%
test-data/allhypo.test: accuracy: 89.61%
test-data/anneal.test: accuracy: 82.0%
test-data/arrhythmia.test: accuracy: 100.0%
test-data/breast-cancer.test: accuracy: 100.0%
test-data/bupa.test: accuracy: 100.0%
test-data/glass.test: accuracy: 100.0%
test-data/iris.test: accuracy: 100.0%
test-data/long.test: accuracy: 100.0%
test-data/parkinsons_updrs.test: accuracy: 100.0%
test-data/soybean-large.test: accuracy: 97.87%
test-data/tic-tac-toe.test: accuracy: 100.0%
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
