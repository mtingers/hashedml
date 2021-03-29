import sys
import re
from collections import Counter
from collections import deque
from random import choice, randint
from textblob import TextBlob

FNV = 14695981039346656037
FNV_PRIME = 1099511628211
FNV_MAX = FNV**7
HSH_P = 211

def find_nearest(array, value):
    """Find closet value in array to 'value' param"""
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return val

def most_common(lst, exclude=None, rand_max_common=4):
    """Find the most common items in a list"""
    counts = Counter(lst)
    if exclude:
        mc = [i[0] for i in counts.most_common(20)]
        for _ in range(30):
            c = choice(mc)
            if not c in exclude:
                return c
        return choice(mc)
    else:
        mc = counts.most_common(randint(1, rand_max_common))[0][0]
        return mc

class HashedML:
    def __init__(self, nback=4):
        self.nback = nback
        self.hmap = {}
        self._tests = 0
        self._tests_correct = 0
        self._accuracy = 0.0
        self._stm = deque(maxlen=20)
        self._stm_last = 0

    def _hashit(self, X):
        hsh_end = 0
        for i, x in enumerate(X):
            hsh = FNV_PRIME
            for c in x:
                if hsh < FNV_MAX:
                    hsh += ord(c)
                    hsh *= HSH_P
                    hsh *= 1 + (i*hsh)
                else:
                    hsh *= ord(c)
                    hsh += HSH_P + i
            hsh_end += hsh
        return hsh_end

    def _fit(self, X, y):
        y = y #str(y)
        X = [str(i) for i in X]
        for i in X:
            h = self._hashit(X)
            if not h in self.hmap:
                self.hmap[h] = []
            if not y in self.hmap[h]:
                self.hmap[h].append(y)

    def fit(self, X, y):
        if not isinstance(X[0], list):
            X = [X]
            y = [y]
        for i in range(len(X)):
            self._fit(X[i], y[i])

    def _predict(self, X, return_one=True):
        X = [str(i) for i in X]
        h = self._hashit(X)
        nearest = find_nearest(list(self.hmap.keys()), h)
        if return_one:
            prediction = most_common(self.hmap[nearest])
        else:
            prediction = []
            print(self.hmap[nearest])
            counts = Counter(self.hmap[nearest])
            top10 = counts.most_common(10)
            for i in top10:
                prediction.append(i[0])
        return prediction

    def predict(self, X, return_one=True):
        predictions = []
        if not isinstance(X[0], list):
            X = [X]
        for x in X:
            predictions.append(self._predict(x, return_one=return_one))
        return predictions

    def test(self, X, y):
        predictions = []
        if not isinstance(X[0], list):
            X = [X]
            y = [y]
        predictions = self.predict(X, return_one=True)
        for i in range(len(X)):
            if predictions[i] == y[i]:
                self._tests_correct += 1
            self._tests += 1
            self._accuracy = round(self._tests_correct/self._tests, 4)
        return predictions

    def accuracy(self):
        return self._accuracy

    def generate(self, X, nwords=100, stm=True, separator=' '):
        if isinstance(X[0], list):
            raise Exception('generate() only supports X of 1 dimension')
        output = ' '.join([str(i) for i in X])+' '
        prev = ''
        prevs = deque(maxlen=self.nback)
        for _ in range(nwords):
            h = self._hashit(X)
            nearest = find_nearest(list(self.hmap.keys()), h)
            if stm:
                n_items = len(set(self.hmap[nearest]))
                # keyword extraction
                if n_items < 3:
                    self._stm.append(nearest)
                    guess = most_common(self.hmap[nearest], exclude=list(prevs))
                    self._stm_last += 1
                elif self._stm_last > 15:
                    found = False
                    for _ in range(10):
                        X2 = X[0:-1]
                        X2.append(most_common(self.hmap[nearest]))
                        h = self._hashit(X)
                        nearest = find_nearest(list(self.hmap.keys()), h)
                        n_items = len(set(self.hmap[nearest]))
                        if n_items < 4:
                            found = True
                            break
                    if not found:
                        h = self._hashit(X)
                        nearest = find_nearest(list(self.hmap.keys()), h)
                    guess = most_common(self.hmap[nearest], exclude=list(prevs))
                    self._stm_last = 0
                else:
                    guess = most_common(self.hmap[nearest], exclude=list(prevs))
                    self._stm_last += 1
            else:
                guess = most_common(self.hmap[nearest], exclude=list(prevs))
            X.append(guess)
            if len(X)+1 > self.nback:
                X = X[1:]
            if not separator in guess:
                guess = '{}{}'.format(guess, separator)
            output_tmp = '{}{}'.format(output, guess)
            tb = TextBlob(output_tmp)
            ngrams = tb.ngrams(n=self.nback*2)
            if len(ngrams) > 2:
                last4 = ngrams[-1]
                if not last4 in ngrams[:-1]:
                    output = output_tmp
            else:
                output = output_tmp
            if not guess in ('\n', ' '):
                prevs.append(guess)
        return output

    def dump_map(self):
        from pprint import pprint
        pprint(self.hmap)


def _main_classify():
    model = HashedML()
    train_csv = open(sys.argv[2]).read().strip().split('\n')
    test_csv = open(sys.argv[3]).read().strip().split('\n')
    for i in train_csv:
        X = i.split(',')[:-1]
        y = i.split(',')[-1]
        model.fit(X, y)
    for i in test_csv:
        X = i.split(',')[:-1]
        y = i.split(',')[-1]
        p = model.test(X, y)
    print('accuracy: {:.2f}%'.format(model.accuracy()*100))


def _fix_tokens(tokens):
    new_tokens = []
    special = (
        '"', "'", '!', ',', '.', '?', '"', "'",
        ';', ':', '“', '’', '’' '”', '”', '“'
    )
    for i in tokens:
        if new_tokens and i in special:
            new_tokens[-1] = new_tokens[-1]+i
        else:
            new_tokens.append(i)
    return new_tokens

def _main_generate():
    from collections import deque
    model = HashedML(nback=4)
    dq = deque(maxlen=model.nback)
    tokens = []
    for fpath in sys.argv[4:]:
        print('input-file:', fpath)
        ##tokens += re.findall(r"[\w'\"]+|[.,!?;\n]", open(fpath).read())
        #tokens += re.findall(r"\w+|[^\w\s]|\n", open(fpath).read(), re.UNICODE)
        #tokens += re.split(' ', open(fpath).read())
        #try:
        #    tokens += re.findall('\n|\w+|[^a-zA-Z0-9]',
        #       open(fpath).read(), re.UNICODE)
        #except:
        #    print('pass:', fpath)
        try:
            tb = TextBlob(open(fpath).read())
        except Exception as err:
            print(err, fpath)
            continue
        tokens += tb.tokens
    tokens = _fix_tokens(tokens)
    for i in tokens:
        dq.append(i)
        if len(dq) != model.nback:
            continue
        c = list(dq)
        X = c[:-1]
        y = c[-1] #.strip()
        model.fit(X, y)
    output = model.generate(
        sys.argv[3].split(' ')[:model.nback-1],
        nwords=int(sys.argv[2]),
        separator=' ')
    print('output:')
    print(output)

def _usage():
    print('usage:')
    print(' {} <classify|generate> ...'.format(sys.argv[0]))
    print(' {} classify <train-csv> <test-csv>'.format(sys.argv[0]))
    print(' {} classify iris.data iris.test'.format(sys.argv[0]))
    print(' {} generate <nwords> <start> <input-file> [<input-file>] ...'.format(
        sys.argv[0]))
    print(' {} generate 200 "Where are we" input/*txt other/foo.txt'.format(
        sys.argv[0]))
    exit(1)

def main():
    if len(sys.argv) < 4:
        _usage()
    if sys.argv[1] == 'classify':
        _main_classify()
    elif sys.argv[1] == 'generate':
        _main_generate()
    else:
        _usage()

if __name__ == '__main__':
    main()

