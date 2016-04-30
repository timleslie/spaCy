import json
from os import path
from collections import defaultdict
import pathlib

from libc.string cimport memset

from cymem.cymem cimport Pool
from thinc.typedefs cimport atom_t, weight_t
from thinc.extra.eg cimport Example
from thinc.structs cimport ExampleC
from thinc.linear.avgtron cimport AveragedPerceptron
from thinc.linalg cimport VecVec

from .typedefs cimport attr_t
from .tokens.doc cimport Doc
from .attrs cimport TAG
from .parts_of_speech cimport NO_TAG, ADJ, ADV, ADP, CONJ, DET, NOUN, NUM, PRON
from .parts_of_speech cimport VERB, X, PUNCT, EOL, SPACE

from .attrs cimport *

from .util import get_package

 
cpdef enum:
    P2_orth
    P2_cluster
    P2_shape
    P2_prefix
    P2_suffix
    P2_pos
    P2_lemma
    P2_flags

    P1_orth
    P1_cluster
    P1_shape
    P1_prefix
    P1_suffix
    P1_pos
    P1_lemma
    P1_flags

    W_orth
    W_cluster
    W_shape
    W_prefix
    W_suffix
    W_pos
    W_lemma
    W_flags

    N1_orth
    N1_cluster
    N1_shape
    N1_prefix
    N1_suffix
    N1_pos
    N1_lemma
    N1_flags

    N2_orth
    N2_cluster
    N2_shape
    N2_prefix
    N2_suffix
    N2_pos
    N2_lemma
    N2_flags

    N_CONTEXT_FIELDS


cdef class TaggerModel(AveragedPerceptron):
    cdef void set_featuresC(self, ExampleC* eg, const TokenC* tokens, int i) except *:
        
        _fill_from_token(&eg.atoms[P2_orth], &tokens[i-2])
        _fill_from_token(&eg.atoms[P1_orth], &tokens[i-1])
        _fill_from_token(&eg.atoms[W_orth], &tokens[i])
        _fill_from_token(&eg.atoms[N1_orth], &tokens[i+1])
        _fill_from_token(&eg.atoms[N2_orth], &tokens[i+2])

        eg.nr_feat = self.extracter.set_features(eg.features, eg.atoms)


cdef inline void _fill_from_token(atom_t* context, const TokenC* t) nogil:
    context[0] = t.lex.lower
    context[1] = t.lex.cluster
    context[2] = t.lex.shape
    context[3] = t.lex.prefix
    context[4] = t.lex.suffix
    context[5] = t.tag
    context[6] = t.lemma
    if t.lex.flags & (1 << IS_ALPHA):
        context[7] = 1
    elif t.lex.flags & (1 << IS_PUNCT):
        context[7] = 2
    elif t.lex.flags & (1 << LIKE_URL):
        context[7] = 3
    elif t.lex.flags & (1 << LIKE_NUM):
        context[7] = 4
    else:
        context[7] = 0


cdef class Tagger:
    """A part-of-speech tagger for English"""
    @classmethod
    def read_config(cls, data_dir):
        return json.load(open(path.join(data_dir, 'pos', 'config.json')))

    templates = (
        (W_orth,),
        (P1_lemma, P1_pos),
        (P2_lemma, P2_pos),
        (N1_orth,),
        (N2_orth,),

        (W_suffix,),
        (W_prefix,),

        (P1_pos,),
        (P2_pos,),
        (P1_pos, P2_pos),
        (P1_pos, W_orth),
        (P1_suffix,),
        (N1_suffix,),

        (W_shape,),
        (W_cluster,),
        (N1_cluster,),
        (N2_cluster,),
        (P1_cluster,),
        (P2_cluster,),

        (W_flags,),
        (N1_flags,),
        (N2_flags,),
        (P1_flags,),
        (P2_flags,),
    )

    @classmethod
    def load(cls, data_dir, vocab):
        data_dir = pathlib.Path(data_dir)
        with (data_dir / 'templates.json').open() as file_:
            templates = json.load(file_)
        model = TaggerModel(templates)
        model.load(str(data_dir / 'model'))
        return cls(vocab, model)

    def __init__(self, Vocab vocab, TaggerModel model):
        self.vocab = vocab
        self.model = model
        
        # TODO: Move this to tag map
        self.freqs = {TAG: defaultdict(int)}
        #for tag in self.vocab.morphology.tag_names:
        #    self.freqs[TAG][self.vocab.strings[tag]] = 1
        self.freqs[TAG][0] = 1

    def __call__(self, Doc tokens):
        """Apply the tagger, setting the POS tags onto the Doc object.

        Args:
            tokens (Doc): The tokens to be tagged.
        """
        if tokens.length == 0:
            return 0

        cdef Pool mem = Pool()

        cdef int i, tag
        cdef Example eg = Example(nr_atom=N_CONTEXT_FIELDS,
                                  nr_class=self.vocab.morphology.n_tags,
                                  nr_feat=self.model.nr_feat)
        for i in range(tokens.length):
            if tokens.c[i].pos == 0:
                self.model.set_featuresC(&eg.c, tokens.c, i)
                self.model.set_scoresC(eg.c.scores,
                    eg.c.features, eg.c.nr_feat)
                tokens.set_tag(i, eg.guess)
                eg.fill_scores(0, eg.c.nr_class)
        tokens.finalize_tags()

    def pipe(self, stream, batch_size=1000, n_threads=2):
        for doc in stream:
            self(doc)
            yield doc
    
    def train(self, Doc tokens, object gold_tag_strs):
        assert len(tokens) == len(gold_tag_strs)
        tag_names = self.vocab.morphology.tag_names
        for tag in gold_tag_strs:
            if tag != None and tag not in tag_names:
                msg = ("Unrecognized gold tag: %s. tag_map.json must contain all"
                       "gold tags, to maintain coarse-grained mapping.")
                raise ValueError(msg % tag)
        golds = [tag_names.index(g) if g is not None else -1 for g in gold_tag_strs]
        cdef int correct = 0
        cdef Pool mem = Pool()
        cdef Example eg = Example(
            nr_atom=N_CONTEXT_FIELDS,
            nr_class=self.vocab.morphology.n_tags,
            nr_feat=self.model.nr_feat)
        for i in range(tokens.length):
            self.model.set_featuresC(&eg.c, tokens.c, i)
            eg.costs = [ 1 if golds[i] not in (c, -1) else 0 for c in xrange(eg.nr_class) ]
            self.model.set_scoresC(eg.c.scores,
                eg.c.features, eg.c.nr_feat)
            self.model.updateC(&eg.c)

            tokens.set_tag(i, eg.guess)
            
            correct += eg.cost == 0
            self.freqs[TAG][tokens.c[i].tag] += 1
            eg.fill_scores(0, eg.c.nr_class)
            eg.fill_costs(0, eg.c.nr_class)
        tokens.finish_tagging()
        return correct
