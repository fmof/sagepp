#!/usr/bin/env python

import argparse
from collections import defaultdict
from numpy import exp
from numpy.random import multinomial
import random
from scipy.misc import logsumexp
from scipy.special import psi
import sys

def read_corpus(fname):
    corpus = []
    with open(fname) as f:
        for line in f.readlines():
            sline = line.split('#')
            doc = sline[0]
            comment = None
            if len(sline) >= 2:
                comment = sline[1]
            sdoc = doc.split(' ')
            label = sdoc[0]
            recorded_doc = {
                'id': comment,
                'label' : label,
                'words' : [],
                'counts' : []
            }
            for ftval in sdoc[1:len(sdoc)]:
                ftval = ftval.strip()
                if len(ftval) == 0:
                    continue
                (word,count) = ftval.split(':')
                recorded_doc['words'].append(int(word))
                recorded_doc['counts'].append(int(count))
            corpus.append(recorded_doc)
    return corpus

def generate_corpus_vocab(args):
    vocab = Vocab()
    for i in xrange(1,args.num_words+1):
        vocab.add_word('word_' + str(i))
    ## this indicates how many vocab words to allocate for the vocab
    background_v_p = args.background_vocab_prop
    ## find the lengths of the background vs. non-background portions of the vocab
    background_vocab = int(len(vocab) * background_v_p)
    non_back_vocab = len(vocab) - background_vocab
    ## set up the non-background vocab span lengths: note we don't
    ## add an offset of background_vocab here, but we will have to later on
    larger_v_span = int(non_back_vocab * .5)
    smaller_v_span = non_back_vocab - larger_v_span
    ## Now find out how many words per doc to draw
    ## this indicates how many words per document to draw from the vocab
    background_prop = args.generated_background_prop
    back_words_per_doc = int(args.words_per_doc * background_prop)
    nb_words_per_doc = args.words_per_doc - back_words_per_doc
    larger_num = int(nb_words_per_doc * args.gen_bias)
    smaller_num = nb_words_per_doc - larger_num
    ## now set up the probability spans
    background_p = background_vocab * ( [ 1.0/float(background_vocab) ] if background_vocab > 0 else [0.0])
    larger_p = larger_v_span * [ 1.0/float(larger_v_span) ]
    smaller_p = smaller_v_span * [ 1.0/float(smaller_v_span) ]
    left_doc_draw = larger_num
    right_doc_draw = smaller_num
    corpus = []
    for di in xrange(args.num_docs):
        # swap the pointers at the halfway point
        if di == int(args.num_docs/2):
            left_doc_draw = smaller_num
            right_doc_draw = larger_num
        doc = {'id' : 'doc_'+str(di),
               'label' : None,
               'words' : [], 'counts' : []}
        counter = defaultdict(int)
#        print "NEW DOC"
        for (index,count) in enumerate(multinomial(back_words_per_doc, background_p)):
            if count > 0:
                counter[index] = count
#                print "\tbackground: %d (%d)" % (index, count)
        for (index,count) in enumerate(multinomial(left_doc_draw, larger_p)):
            if count > 0:
                counter[index + background_vocab] = count
#                print "\tleft: %d (%d)" % (index + background_vocab, count)
        for (index, count) in enumerate(multinomial(right_doc_draw, smaller_p)):
            if count > 0:
                counter[index + background_vocab + larger_v_span] = count
#                print "\tright: %d (%d)" % (index + background_vocab + larger_v_span, count)
        for (word, count) in counter.iteritems():
            doc['words'].append(word)
            doc['counts'].append(count)
        corpus.append(doc)
    return (corpus, vocab)
    
class Vocab:
    def __init__(self):
        self.oov = "____OOV___"
        self.words = [self.oov]
        self.index = {self.oov : 0}
    def add_word(self, word_str):
        self.words.append(word_str)
        self.index[word_str] = len(self.words)
    def __len__(self):
        return len(self.words)
    
def read_vocab(fname):
    vocab = Vocab()
    with open(fname) as f:
        for line in f.readlines():
            vocab.add_word(line.strip())
    return vocab

class LDAV:
    def __init__(self, num_topics):
        self.num_topics_ = num_topics
        self.var_assignment_params_ = []
        self.var_usage_params_ = []
        self.var_topic_params_ = []
        self.corpus_ = None
        self.vocab_  = None
        self.alpha_  = self.num_topics_ * [0.1]
        self.beta_   = None

    def init(self, corpus, vocab):
        self.corpus_ = corpus
        self.vocab_  = vocab
        self.beta_ = len(vocab) * [0.1]
        iv = 1.0/float(self.num_topics_)
        for doc in corpus:
            self.var_assignment_params_.append([
                [iv + random.uniform(-iv,iv) for k in xrange(self.num_topics_)]
            for w in xrange(len(doc['words']))])
            self.var_usage_params_.append([iv + random.uniform(-iv,iv) for k in xrange(self.num_topics_)])
        iw = 1.0/float(len(vocab))
        for ti in xrange(self.num_topics_):
            self.var_topic_params_.append([iw + random.uniform(-iw, iw) for w in xrange(len(vocab))])

    def update_assignments(self, doc_idx, word, word_idx, psi_gamma_sum, psi_lambda_sum):
        ## propto exp(psi(gamma_{d,k}) - psi(sum_j gamma_{d,k}) + psi(lambda_{k,v}) - psi(sum_w lambda_{k,w}))
        t0 = [psi(self.var_usage_params_[doc_idx][k]) - psi_gamma_sum + psi(self.var_topic_params_[k][word]) - psi_lambda_sum[k] for k in xrange(self.num_topics_)]
        lnorm = logsumexp(t0)
        for k in xrange(self.num_topics_):
            import math
            self.var_assignment_params_[doc_idx][word_idx][k] = exp(t0[k] - lnorm)
            if math.isnan(self.var_assignment_params_[doc_idx][word_idx][k]):
                raise Exception()

    def update_usage(self, doc_id):
        doc = self.corpus_[doc_id]
        for k in xrange(self.num_topics_):
            self.var_usage_params_[doc_id][k] = self.alpha_[k]
        for wid in xrange(len(doc['words'])):
            for k in xrange(self.num_topics_):
                self.var_usage_params_[doc_id][k] += self.var_assignment_params_[doc_id][wid][k] * doc['counts'][wid]
                import math
                if math.isnan(self.var_usage_params_[doc_id][k]):
                    raise Exception()
            
    def e_step(self, e_iter = 1):
        psi_lambda_sum = []
        for k in xrange(self.num_topics_):
            lsum = sum(self.var_topic_params_[k])
            psi_lambda_sum.append(psi(lsum))
        for ei in xrange(e_iter):
            doc_id = 0
            for doc in self.corpus_:
                num_words = len(doc['words'])
                psi_gamma_sum = psi(sum(self.var_usage_params_[doc_id]))
                for word_idx in xrange(num_words):
                    word = doc['words'][word_idx]
                    self.update_assignments(doc_id, word, word_idx, psi_gamma_sum, psi_lambda_sum)
                self.update_usage(doc_id)
                doc_id += 1

    def m_step(self):
        for k in xrange(self.num_topics_):
            self.var_topic_params_[k] = [b for b in self.beta_]
            doc_id = 0
            for doc in self.corpus_:
                num_words = len(doc['words'])
                for word_idx in xrange(num_words):
                    word = doc['words'][word_idx]
                    self.var_topic_params_[k][word] += doc['counts'][word_idx] * self.var_assignment_params_[doc_id][word_idx][k]
                doc_id += 1

    def print_usage(self, fname):
        with open(fname,'w') as f:
            for doc in self.var_usage_params_:
                norm = sum(doc)
                n1 = [str(x/norm) for x in doc]
                f.write(' '.join(n1))
                f.write('\n')

    def print_topics(self, fname):
        with open(fname,'w') as f:
            for topic in self.var_topic_params_:
                norm = sum(topic)
                n1 = [str(x/norm) for x in topic]
                f.write(' '.join(n1))
                f.write('\n')
                
    def learn(self, args):
        for i in xrange(args.em_iters):
            self.e_step(args.e_iters)
            self.m_step()
            if args.print_usage:
                self.print_usage(args.usage_file + str(i))
            if args.print_topics:
                self.print_topics(args.topic_file + str(i))


def main(args):
    vocab = read_vocab(args.vocab_file)
    corpus = read_corpus(args.corpus_file)
    ldav = LDAV(args.num_topics)
    ldav.init(corpus, vocab)
    ldav.learn(args)

def generate_main(args):
    (corpus, vocab) = generate_corpus_vocab(args)
    if args.print_generated:
        vf = sys.stdout if args.generated_vocab_name == '-' else open(args.generated_vocab_name, 'w')
        for word in vocab.words:
            vf.write(word)
            vf.write('\n')
        if not(args.generated_vocab_name == '-'):
            vf.close()
        ### now print corpus
        cf = sys.stdout if args.generated_corpus_name == '-' else open(args.generated_corpus_name, 'w')
        for doc in corpus:
            label = '0' if doc['label'] == None else str(doc['label'])
            cf.write(label)
            cf.write(' ')
            for wid in xrange(len(doc['words'])):
                cf.write(str(doc['words'][wid]) + ':' + str(doc['counts'][wid]) + ' ')
            cf.write('# ' + str(doc['id']))
            cf.write('\n')
        if not(args.generated_corpus_name == '-'):
            cf.close()
                
    if args.infer_generated:
        ldav = LDAV(args.num_topics)
        ldav.init(corpus, vocab)
        ldav.learn(args.em_iters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab',dest = 'vocab_file')
    parser.add_argument('--corpus', dest = 'corpus_file')
    parser.add_argument('--topics', dest = 'num_topics', type=int, default = 10)
    parser.add_argument('--em-iters',dest='em_iters',type=int, default = 10)
    parser.add_argument('--e-iters',dest='e_iters',type=int, default = 1)
    parser.add_argument('--num-docs',dest='num_docs',type=int, default = 10)
    parser.add_argument('--gen-bias',dest='gen_bias',type=float, default = 0.8)
    parser.add_argument('--num-words',dest='num_words',type=int, default = 10)
    parser.add_argument('--words-per-doc', dest = 'words_per_doc', type = int, default = 10)
    parser.add_argument('--print-usage', dest = 'print_usage', default = False, action = 'store_true')
    parser.add_argument('--print-topics', dest = 'print_topics', default = False, action = 'store_true')
    parser.add_argument('--usage-file', dest = 'usage_file', type = str, default = 'py_usage_')
    parser.add_argument('--topic-file', dest = 'topic_file', type = str, default = 'py_topics_')
    parser.add_argument('--generate', dest='generate', default = False, action = 'store_true')
    parser.add_argument('--generated-background', dest='generated_background_prop', default = 0.0, type = float)
    parser.add_argument('--generated-background-vocab-prop', dest='background_vocab_prop', default = 0.0, type = float)
    parser.add_argument('--no-infer-generated',dest='infer_generated',default=True, action = 'store_false')
    parser.add_argument('--print-generated', dest='print_generated',default=False, action = 'store_true')
    parser.add_argument('--generated-vocab-name', dest='generated_vocab_name',type=str)
    parser.add_argument('--generated-corpus-name',dest='generated_corpus_name',type=str)
    args = parser.parse_args()
    if args.generate:
        generate_main(args)
    else:
        main(args)
                

            
