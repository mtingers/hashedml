# HashedML
A machine learning library that uses a different approach: string hashing
(think hash tables) for classifying sequences.

# Installation

```
pip install -U hashedml
```

# Classification
HashedML takes the simple `fit(X, y)` / `predict(X)` approach.

Example:

```python
model = HashedML()
iris_data = open('iris.data').read().split('\n')
for i in iris_data:
    X = i[:-1]
    y = i[-1]
    model.fit(X, y)

iris_test = open('iris.test').read().split('\n')
for i in irist_test:
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

```
from collections import deque
model = HashedML(nback=4, stm=True)
token_q = deque(maxlen=model.nback)
tokens = []

tokens = TextBlob(open('training.text').read()).tokens

# Learn
for i in tokens:
    token_q.append(i)
    if len(token_q) != model.nback:
        continue
    X = list(token_q)tq[:-1]
    y = list(token_q)tq[-1]
    model.fit(X, y)

# Generate
output = model.generate(
    ('What', 'is'),
    nwords=500,
    seperator=' '
)
print(output)
```

# Variable X Input & Non-numerical X or Y
The X value can be of varying length/dimensions. For example, this is valid:
```
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
