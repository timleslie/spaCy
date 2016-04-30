from __future__ import absolute_import
from os import path
from warnings import warn
import io
import pathlib

try:
    import ujson as json
except ImportError:
    import json

from .tokenizer import Tokenizer
from .vocab import Vocab
from .syntax.parser import Parser
from .syntax.arc_eager import ArcEager
from .syntax.ner import BiluoPushDown
from .tagger import Tagger
from .matcher import Matcher
from .serialize.packer import Packer
from . import attrs
from . import orth

from . import util
from . import about
from .attrs import TAG, DEP, ENT_IOB, ENT_TYPE, HEAD


class Language(object):
    lang = None

    def __init__(self,
        data_dir=None,
        vocab=None,
        tokenizer=None,
        pipeline=None,
        vectors=None):
        '''
        Create a text-processing pipeline
        '''
        if data_dir is None:
            data_dir = util.get_package_by_name(about.__models__[self.lang])
        data_dir = pathlib.Path(data_dir)

        self.vectors = self.load_vectors(data_dir) if vectors in (None, True) else vectors
        self.vocab = self.load_vocab(data_dir) if vocab in (None, True) else vocab
        self.tokenizer = self.load_tokenizer(data_dir) \
                            if tokenizer in (None, True) else tokenizer
        self.pipeline = self.load_pipeline(data_dir) \
                            if pipeline in (None, True) else pipeline

    def load_vectors(self, data_dir):
        data_dir = pathlib.Path(data_dir) / 'vectors'
        return VectorStore.load(data_dir)

    def load_vocab(self, data_dir):
        data_dir = pathlib.Path(data_dir) / 'vocab'
        return Vocab.load(data_dir, self.vectors)
    
    def load_tokenizer(self, data_dir):
        data_dir = pathlib.Path(data_dir) / 'tokenizer'
        return Tokenizer.load(data_dir, self.vocab)

    def load_pipeline(self, data_dir):
        data_dir = pathlib.Path(data_dir)
        return (self.load_tagger(data_dir), self.load_matcher(data_dir),
                self.load_parser(data_dir), self.load_entity(data_dir))

    def load_tagger(self, data_dir):
        data_dir = pathlib.Path(data_dir) / 'pos'
        return Tagger.load(data_dir, self.vocab)

    def load_parser(self, data_dir):
        data_dir = pathlib.Path(data_dir) / 'deps'
        return Parser.load(data_dir, self.vocab.strings, ArcEager)
    
    def load_entity(self, data_dir):
        data_dir = pathlib.Path(data_dir) / 'ner'
        return Parser.load(data_dir, self.vocab.strings, BiluoPushDown)
    
    def load_matcher(self, data_dir):
        loc = pathlib.Path(data_dir) / 'vocab' / 'gazetteer.json'
        return Matcher.load(loc, self.vocab)

    def __call__(self, text, tag=True, parse=True, entity=True):
        """Apply the pipeline to some text.  The text can span multiple sentences,
        and can contain arbtrary whitespace.  Alignment into the original string
        is preserved.
        
        Args:
            text (unicode): The text to be processed.

        Returns:
            tokens (spacy.tokens.Doc):

        >>> from spacy.en import English
        >>> nlp = English()
        >>> tokens = nlp('An example sentence. Another example sentence.')
        >>> tokens[0].orth_, tokens[0].head.tag_
        ('An', 'NN')
        """
        tokens = self.tokenizer(text)
        skip = {self.tagger: not tag, self.parser: not parse, self.entity: not entity}
        for annotator in self.pipeline:
            if not tag and annotator is self.tagger:
                continue
            elif not parse and annotator is self.parser:
                continue
            elif not entity and annotator is self.entity:
                continue
            if annotator is not None:
                annotator(tokens)
        return annotator

    def pipe(self, texts, tag=True, parse=True, entity=True, n_threads=2,
            batch_size=1000):
        stream = self.tokenizer.pipe(texts,
            n_threads=n_threads, batch_size=batch_size)
        skip = {self.tagger: not tag, self.parser: not parse, self.entity: not entity}
        for annotator in self.pipeline:
            if skip.get(annotator):
                continue
            if hasattr(annotator, 'pipe'):
                stream = annotator.pipe(stream, n_threads=n_threads, batch_size=batch_size)
            else:
                stream = (annotator(doc) for doc in stream)
        for doc in stream:
            yield doc

    def end_training(self, data_dir=None):
        if data_dir is None:
            data_dir = self.data_dir
        if self.parser:
            self.parser.model.end_training()
            self.parser.model.dump(path.join(data_dir, 'deps', 'model'))
        if self.entity:
            self.entity.model.end_training()
            self.entity.model.dump(path.join(data_dir, 'ner', 'model'))
        if self.tagger:
            self.tagger.model.end_training()
            self.tagger.model.dump(path.join(data_dir, 'pos', 'model'))

        strings_loc = path.join(data_dir, 'vocab', 'strings.json')
        with io.open(strings_loc, 'w', encoding='utf8') as file_:
            self.vocab.strings.dump(file_)
        self.vocab.dump(path.join(data_dir, 'vocab', 'lexemes.bin'))

        if self.tagger:
            tagger_freqs = list(self.tagger.freqs[TAG].items())
        else:
            tagger_freqs = []
        if self.parser:
            dep_freqs = list(self.parser.moves.freqs[DEP].items())
            head_freqs = list(self.parser.moves.freqs[HEAD].items())
        else:
            dep_freqs = []
            head_freqs = []
        if self.entity:
            entity_iob_freqs = list(self.entity.moves.freqs[ENT_IOB].items())
            entity_type_freqs = list(self.entity.moves.freqs[ENT_TYPE].items())
        else:
            entity_iob_freqs = []
            entity_type_freqs = []
        with open(path.join(data_dir, 'vocab', 'serializer.json'), 'w') as file_:
            file_.write(
                json.dumps([
                    (TAG, tagger_freqs),
                    (DEP, dep_freqs),
                    (ENT_IOB, entity_iob_freqs),
                    (ENT_TYPE, entity_type_freqs),
                    (HEAD, head_freqs)
                ]))

#
#    @classmethod
#    def default_vocab(cls, package, get_lex_attr=None, vectors_package=None):
#        if get_lex_attr is None:
#            if package.has_file('vocab', 'oov_prob'):
#                with package.open(('vocab', 'oov_prob')) as file_:
#                    oov_prob = float(file_.read().strip())
#                get_lex_attr = cls.default_lex_attrs(oov_prob=oov_prob)
#            else:
#                get_lex_attr = cls.default_lex_attrs()
#        if hasattr(package, 'dir_path'):
#            return Vocab.from_package(package, get_lex_attr=get_lex_attr,
#                vectors_package=vectors_package)
#        else:
#            return Vocab.load(package, get_lex_attr)
#
#    @classmethod
#    def default_parser(cls, package, vocab):
#        if hasattr(package, 'dir_path'):
#            data_dir = package.dir_path('deps')
#        else:
#            data_dir = package
#        if data_dir and path.exists(data_dir):
#            return Parser.from_dir(data_dir, vocab.strings, ArcEager)
#        else:
#            return None
#
#    @classmethod
#    def default_entity(cls, package, vocab):
#        if hasattr(package, 'dir_path'):
#            data_dir = package.dir_path('ner')
#        else:
#            data_dir = package
#        if data_dir and path.exists(data_dir):
#            return Parser.from_dir(data_dir, vocab.strings, BiluoPushDown)
#        else:
#            return None
#
#
