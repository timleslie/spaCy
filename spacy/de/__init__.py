from __future__ import unicode_literals, print_function

from os import path

from ..language import Language
from ..syntax.nonproj import PseudoProjectivity


class German(Language):
    lang = 'de'

    def load_tokenizer(self, data_dir):
        data_dir = pathlib.Path(data_dir) / 'tokenizer'
        return GermanTokenizer.load(data_dir, self.vocab)

    def load_pipeline(self, data_dir):
        data_dir = pathlib.Path(data_dir)
        return (self.load_tagger(data_dir), self.load_matcher(data_dir),
                self.load_parser(data_dir), PseudoProjectivity.deprojectivize,
                self.load_entity(data_dir))
