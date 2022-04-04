import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict

class FreqDist(Counter):
    """
    Stuff here.
    """

    def __init__(self, samples=None):
        """
        Construct a new frequency distribution.  If ``samples`` is
        given, then the frequency distribution will be initialized
        with the count of each object in ``samples``; otherwise, it
        will be initialized to be empty.

        In particular, ``FreqDist()`` returns an empty frequency
        distribution; and ``FreqDist(samples)`` first creates an empty
        frequency distribution, and then calls ``update`` with the
        list ``samples``.

        :param samples: The samples to initialize the frequency
            distribution with.
        :type samples: Sequence
        """
        Counter.__init__(self, samples)

        # Cached number of samples in this FreqDist
        self._N = None


    def N(self):
        """
        Return the total number of sample outcomes that have been
        recorded by this FreqDist.  For the number of unique
        sample values (or bins) with counts greater than zero, use
        ``FreqDist.B()``.

        :rtype: int
        """
        if self._N is None:
            # Not already cached, or cache has been invalidated
            self._N = sum(self.values())
        return self._N
    

class NBGenderClassifier():
    def __init__(self, smoothing_alpha):
        self.alpha = smoothing_alpha # "smoothing" parameter
    

    def update_feature_counts(self, name, gender, p):
        
        for f in self.get_features(name):
            if f not in p:
                p[f] = 1 + self.alpha
            else:
                p[f] += 1
            
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
        label_probdist = estimator(label_freqdist)

        # Create the P(fval|label, fname) distribution
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label, fname] = probdist

        return cls(label_probdist, feature_probdist)
        
        
        
        # self.names = names
        # self.gender_labels = gender_labels
        
        # # Compute log prior
        # self.n_total = gender_labels.shape[0]
        # self.n_male = sum((gender_labels == "M").astype(int))
        # self.n_female = sum((gender_labels == "F").astype(int))
        
        # self.n_total = gender_labels.shape[0]
        # assert self.n_total == self.n_male + self.n_female
        # self.log_prior = {"M": np.log(self.n_male/self.n_total), 
        #          "F": np.log(self.n_female/self.n_total)}
        
        # self.p_x_given_male = {}
        # self.p_x_given_female = {}
        
        # # TODO: precompute the number of "features", like the number of time the dice was rolled for both male and female cases
        # # Or maybe I can somehow update this on the fly? I kind of doubt that though. 
        # # precompute some normalizing constants?
        # self.n_male_symbols = 0
        # self.n_female_symbols = 0
        # for name, gender in zip(names, gender_labels):
        #     for f in self.get_features(name):
        #         if gender == "M":
        #             self.n_male_symbols += 1
        #         else:
        #             self.n_female_symbols += 1
        
        # # training
        # for name, gender in zip(names, gender_labels):
            
        #     if gender == "M":
        #         p = self.p_x_given_male
        #     else:
        #         p = self.p_x_given_female
            
        #     name = name.lower()
        #     self.update_feature_counts(name, gender, p)

    def compute_prob(self, name, gender):
        if gender == "M":
            prob = self.log_prior[gender]
        else:
            prob = self.log_prior["F"]

    def predict(self, name):
        prob_M_and_F = {}
        for g in ["M", "F"]:
            prob = self.log_prior[g]
            if g == "M":
                n_gender = self.n_male_symbols
                p = self.p_x_given_male
            else:
                n_gender = self.n_female_symbols
                p = self.p_x_given_female
                
        
            for f in self.get_features(name):
                if f not in p:
                    prob += np.log(self.alpha/n_gender)
                else:
                    prob_next = p[f]/n_gender
                    prob += np.log(prob_next)
            
            prob_M_and_F[g] = prob
        
        # print(prob_M_and_F)
        # print(self.log_prior)
        
        if prob_M_and_F["M"] > prob_M_and_F["F"]:
            return "M"
        else:
            return "F"

def get_features(name):
    """
    Take in a single name and return the relevant features. 
    """
    
    feature_dict = {}
    
    # # bigrams
    # s1 = name[:-1]
    # s2 = name[1:]
    # for c1, c2 in zip(s1, s2):
    #     yield c1 + c2
        
    # name endings
    for i in range(1,3):
        if i <= len(name):
            feature_dict[f'letter {-i}'] = name[-i]
    
    
    # count vowels
    n_vowels = sum(map(name.lower().count, "aeiou"))
    feature_dict['n_vowels'] = n_vowels
    
    
    # # name beginnings
    # for i in range(1,1):
    #     if i <= len(name):
    #         beginning = name[:i] + "_begin"
    #         yield beginning
    
    return feature_dict
        
if __name__ == "__main__":
    
    alpha = 1 # use +alpha smoothing
    train_pct = .9

    df = pd.read_csv ('data/name_gender_dataset.csv')

    names = df['Name'].to_numpy()

    labels = df['Gender'].to_numpy()
    
    labeled_names = [(n, label) for n, label in zip(names, labels)]
    random.shuffle(labeled_names) 
    
    featuresets = [(get_features(n), gender) for (n, gender) in labeled_names]

    n_examples = names.shape[0]    
    n_split = int(train_pct*n_examples)
    
    
    train_set, test_set = featuresets[:n_split], featuresets[n_split:]
    
    clf = NBGenderClassifier(smoothing_alpha = alpha)
    clf.train(train_set)
    
    # test
    n_test = 0
    n_correct = 0
    for name, label in zip(names, labels):
        n_test += 1
        y_hat = clf.predict(name)
        if y_hat == label:
            n_correct += 1
    
    print(f'test acc: {n_correct/n_test}')

