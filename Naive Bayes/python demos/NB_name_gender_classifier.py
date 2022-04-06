import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict
import string

class FreqDist(Counter):
    """
    Stuff here.
    """

    def __init__(self, samples=None):

        Counter.__init__(self, samples)

        # Cached number of samples in this FreqDist
        self._N = None


    def N(self):
        if self._N is None:
            # Not already cached, or cache has been invalidated
            self._N = sum(self.values())
        return self._N
    

class NBGenderClassifier():
    def __init__(self):
        pass

    # def update_feature_counts(self, name, gender, p):
        
    #     for f in self.get_features(name):
    #         if f not in p:
    #             p[f] = 1 + self.alpha
    #         else:
    #             p[f] += 1
            
    def train(self, labeled_featuresets):
        
        label_freqdist = FreqDist()
        feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        fnames = set()

        # Count up how many times each feature value occurred, given
        # the label and featurename.
        for featureset, label in labeled_featuresets:
            label_freqdist[label] += 1
            for fname, fval in featureset.items():
                # Increment freq(fval|label, fname)
                feature_freqdist[label, fname][fval] += 1
                # Record that fname can take the value fval.
                feature_values[fname].add(fval)
                # Keep a list of all feature names.
                fnames.add(fname)

        # If a feature didn't have a value given for an instance, then
        # we assume that it gets the implicit value 'None.'  This loop
        # counts up the number of 'missing' feature values for each
        # (label,fname) pair, and increments the count of the fval
        # 'None' by that amount.
        for label in label_freqdist:
            num_samples = label_freqdist[label]
            for fname in fnames:
                count = feature_freqdist[label, fname].N()
                # Only add a None key when necessary, i.e. if there are
                # any samples with feature 'fname' missing.
                if num_samples - count > 0:
                    feature_freqdist[label, fname][None] += num_samples - count
                    feature_values[fname].add(None)

        # Create the P(label) distribution
        label_probdist = {}
        for label in label_freqdist:
            label_probdist[label] = label_freqdist[label]/label_freqdist.N()
            
        # label_probdist = estimator(label_freqdist)

        # Create the P(fval|label, fname) distribution
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = {}
            for fval in freqdist:
                if freqdist.N() == 0:
                    probdist[fval] = 0
                else:
                    probdist[fval] = freqdist[fval]/freqdist.N()
            # probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label, fname] = probdist
        
        
        self._label_probdist = label_probdist
        self._feature_probdist = feature_probdist
        self._labels = list(label_probdist.keys())
        # self._labels = list(label_probdist.samples()) # This line shouldn't work
        
        print('Training Complete...')
        
    def predict(self, featureset):
        
        # Discard any feature names that we've never seen before.
        # Otherwise, we'll just assign a probability of 0 to
        # everything.
        featureset = featureset.copy()
        for fname in list(featureset.keys()):
            for label in self._labels:
                if (label, fname) in self._feature_probdist:
                    break
            else:
                # print('Ignoring unseen feature %s' % fname)
                del featureset[fname]
    
        # Find the log probability of each label, given the features.
        # Start with the log probability of the label itself.
        logprob = {}
        for label in self._labels:
            logprob[label] = np.log(self._label_probdist[label])
    
        # Then add in the log probability of features given labels.
        for label in self._labels:
            for (fname, fval) in featureset.items():
                if (label, fname) in self._feature_probdist:
                    feature_probs = self._feature_probdist[label, fname]
                    if fval not in feature_probs:
                        logprob[label] += np.log(1/40)
                    else:
                        logprob[label] += np.log(feature_probs[fval])

                else:
                    pass

        return max(logprob.keys(), key = lambda x: logprob[x])

def get_features(name):
    """
    Take in a single name and return the relevant features. 
    """
    
    feature_dict = {}
    
        
    # suffix
    for i in range(1,4):
        if i <= len(name):
            # feature_dict[f'letter {-i}'] = name[-i]
            feature_dict[f'suffux {-i}'] = name[-i:]
            
    # first letter
    feature_dict['first letter'] = name[0]
    
    # letter counts
    for letter in string.ascii_lowercase:
        feature_dict[f"count({letter})"] = name.lower().count(letter)
        feature_dict[f"has({letter})"] = (letter in name.lower())
    
    # # bigrams
    # s1 = name[:-1]
    # s2 = name[1:]
    # for c1, c2 in zip(s1, s2):
    #     feature_dict[f'has {c1+c2}'] = c1 + c2 in name
    
    # # count vowels
    # n_vowels = sum(map(name.lower().count, "aeiou"))
    # feature_dict['n_vowels'] = n_vowels
    
    # # name beginnings
    # for i in range(1,1):
    #     if i <= len(name):
    #         beginning = name[:i] + "_begin"
    #         yield beginning
    
    return feature_dict
        
if __name__ == "__main__":
    
    alpha = 1 # use +alpha smoothing
    train_pct = .9

    df = pd.read_csv('data/name_gender_dataset.csv')

    names = df['Name'].to_numpy()

    labels = df['Gender'].to_numpy()
    
    labeled_names = [(n.lower(), label) for n, label in zip(names, labels)]
    random.shuffle(labeled_names) 
    
    featuresets = [(get_features(n), gender) for (n, gender) in labeled_names]

    n_examples = names.shape[0]    
    n_split = int(train_pct*n_examples)
    
    train_set, test_set = featuresets[:n_split], featuresets[n_split:]
    
    clf = NBGenderClassifier()
    clf.train(train_set)

    # test
    n_test = 0
    n_correct = 0
    for featureset, label in test_set:
        n_test += 1
        y_hat = clf.predict(featureset)
        if y_hat == label:
            n_correct += 1
    
    print(f'test acc: {n_correct/n_test}')
    
    