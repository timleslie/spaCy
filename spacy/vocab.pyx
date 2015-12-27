from __future__ import unicode_literals

from libc.stdio cimport fopen, fclose, fread, fwrite, FILE
from libc.string cimport memset
from libc.stdint cimport int32_t
from libc.stdint cimport uint64_t

import bz2
from os import path
import io
import math
import json
import tempfile

from .lexeme cimport EMPTY_LEXEME
from .lexeme cimport Lexeme
from .strings cimport hash_string
from .orth cimport word_shape
from .typedefs cimport attr_t, flags_t
from .cfile cimport CFile
from .lemmatizer import Lemmatizer

from . import attrs
from . import symbols

from cymem.cymem cimport Address
from . import util
from .serialize.packer cimport Packer
from .attrs cimport PROB

try:
    import copy_reg
except ImportError:
    import copyreg as copy_reg


DEF MAX_VEC_SIZE = 100000


cdef float[MAX_VEC_SIZE] EMPTY_VEC
memset(EMPTY_VEC, 0, sizeof(EMPTY_VEC))
memset(&EMPTY_LEXEME, 0, sizeof(LexemeC))
EMPTY_LEXEME.vector = EMPTY_VEC


cdef class Vocab:
    '''A map container for a language's LexemeC structs.
    '''
    @classmethod
    def from_package(cls, package, get_lex_attr=None):
        tag_map = package.load_utf8(json.load,
            'vocab', 'tag_map.json')

        lemmatizer = Lemmatizer.from_package(package)

        serializer_freqs = package.load_utf8(json.load,
            'vocab', 'serializer.json',
            require=False)  # TODO: really optional?

        cdef Vocab self = cls(get_lex_attr=get_lex_attr, tag_map=tag_map,
                              lemmatizer=lemmatizer, serializer_freqs=serializer_freqs)

        if package.has_file('vocab', 'strings.json'):  # TODO: really optional?
            package.load_utf8(self.strings.load, 'vocab', 'strings.json')
            self.load_lexemes(package.file_path('vocab', 'lexemes.bin'))

        if package.has_file('vocab', 'vec.bin'):  # TODO: really optional?
            self.vectors_length = self.load_vectors_from_bin_loc(
                package.file_path('vocab', 'vec.bin'))

        return self

    def __init__(self, get_lex_attr=None, tag_map=None, lemmatizer=None, serializer_freqs=None):
        if tag_map is None:
            tag_map = {}
        if lemmatizer is None:
            lemmatizer = Lemmatizer({}, {}, {})
        self.mem = Pool()
        self._by_hash = PreshMap()
        self._by_orth = PreshMap()
        self.strings = StringStore()
        # Load strings in a special order, so that we have an onset number for
        # the vocabulary. This way, when words are added in order, the orth ID
        # is the frequency rank of the word, plus a certain offset. The structural
        # strings are loaded first, because the vocab is open-class, and these
        # symbols are closed class.
        for name in symbols.NAMES + list(sorted(tag_map.keys())):
            if name:
                _ = self.strings[name]
        self.get_lex_attr = get_lex_attr
        self.morphology = Morphology(self.strings, tag_map, lemmatizer)
        self.serializer_freqs = serializer_freqs
        
        self.length = 1
        self._serializer = None
    
    property serializer:
        def __get__(self):
            if self._serializer is None:
                freqs = []
                self._serializer = Packer(self, self.serializer_freqs)
            return self._serializer

    def __len__(self):
        """The current number of lexemes stored."""
        return self.length

    def __reduce__(self):
        tmp_dir = tempfile.mkdtemp()
        lex_loc = path.join(tmp_dir, 'lexemes.bin')
        str_loc = path.join(tmp_dir, 'strings.json')
        vec_loc = path.join(tmp_dir, 'vec.bin')

        self.dump(lex_loc)
        with io.open(str_loc, 'w', encoding='utf8') as file_:
            self.strings.dump(file_)

        self.dump_vectors(vec_loc)
        
        state = (str_loc, lex_loc, vec_loc, self.morphology, self.get_lex_attr,
                 self.serializer_freqs, self.data_dir)
        return (unpickle_vocab, state, None, None)

    cdef const LexemeC* get(self, Pool mem, unicode string) except NULL:
        '''Get a pointer to a LexemeC from the lexicon, creating a new Lexeme
        if necessary, using memory acquired from the given pool.  If the pool
        is the lexicon's own memory, the lexeme is saved in the lexicon.'''
        if string == u'':
            return &EMPTY_LEXEME
        cdef LexemeC* lex
        cdef hash_t key = hash_string(string)
        lex = <LexemeC*>self._by_hash.get(key)
        cdef size_t addr
        if lex != NULL:
            if lex.orth != self.strings[string]:
                raise LookupError.mismatched_strings(
                    lex.orth, self.strings[string], self.strings[lex.orth], string)
            return lex
        else:
            return self._new_lexeme(mem, string)

    cdef const LexemeC* get_by_orth(self, Pool mem, attr_t orth) except NULL:
        '''Get a pointer to a LexemeC from the lexicon, creating a new Lexeme
        if necessary, using memory acquired from the given pool.  If the pool
        is the lexicon's own memory, the lexeme is saved in the lexicon.'''
        if orth == 0:
            return &EMPTY_LEXEME
        cdef LexemeC* lex
        lex = <LexemeC*>self._by_orth.get(orth)
        if lex != NULL:
            return lex
        else:
            return self._new_lexeme(mem, self.strings[orth])

    cdef const LexemeC* _new_lexeme(self, Pool mem, unicode string) except NULL:
        cdef hash_t key
        cdef bint is_oov = mem is not self.mem
        if len(string) < 3:
            mem = self.mem
        lex = <LexemeC*>mem.alloc(sizeof(LexemeC), 1)
        lex.orth = self.strings[string]
        lex.length = len(string)
        lex.id = self.length
        lex.vector = <float*>mem.alloc(self.vectors_length, sizeof(float))
        if self.get_lex_attr is not None:
            for attr, func in self.get_lex_attr.items():
                value = func(string)
                if isinstance(value, unicode):
                    value = self.strings[value]
                if attr == PROB:
                    lex.prob = value
                else:
                    Lexeme.set_struct_attr(lex, attr, value)
        if is_oov:
            lex.id = 0
        else:
            key = hash_string(string)
            self._add_lex_to_vocab(key, lex)
        assert lex != NULL, string
        return lex

    cdef int _add_lex_to_vocab(self, hash_t key, const LexemeC* lex) except -1:
        self._by_hash.set(key, <void*>lex)
        self._by_orth.set(lex.orth, <void*>lex)
        self.length += 1

    def __iter__(self):
        cdef attr_t orth
        cdef size_t addr
        for orth, addr in self._by_orth.items():
            yield Lexeme(self, orth)

    def __getitem__(self,  id_or_string):
        '''Retrieve a lexeme, given an int ID or a unicode string.  If a previously
        unseen unicode string is given, a new lexeme is created and stored.

        Args:
            id_or_string (int or unicode):
              The integer ID of a word, or its unicode string.  If an int >= Lexicon.size,
              IndexError is raised. If id_or_string is neither an int nor a unicode string,
              ValueError is raised.

        Returns:
            lexeme (Lexeme):
              An instance of the Lexeme Python class, with data copied on
              instantiation.
        '''
        cdef attr_t orth
        if type(id_or_string) == unicode:
            orth = self.strings[id_or_string]
        else:
            orth = id_or_string
        return Lexeme(self, orth)

    cdef const TokenC* make_fused_token(self, substrings) except NULL:
        cdef int i
        tokens = <TokenC*>self.mem.alloc(len(substrings) + 1, sizeof(TokenC))
        for i, props in enumerate(substrings):
            token = &tokens[i]
            # Set the special tokens up to have morphology and lemmas if
            # specified, otherwise use the part-of-speech tag (if specified)
            token.lex = <LexemeC*>self.get(self.mem, props['F'])
            if 'pos' in props:
                self.morphology.assign_tag(token, props['pos'])
            if 'L' in props:
                tokens[i].lemma = self.strings[props['L']]
            for feature, value in props.get('morph', {}).items():
                self.morphology.assign_feature(&token.morph, feature, value)
        return tokens
    
    def dump(self, loc):
        if path.exists(loc):
            assert not path.isdir(loc)
        cdef bytes bytes_loc = loc.encode('utf8') if type(loc) == unicode else loc


        check_sizes()
        
        # Allocate a temporary buffer, with the lexemes. This prevents us from
        # repeated iterating over the hash table.
        cdef Pool tmp_mem = Pool()
        # Allocate memory for the lexemes
        # Note that this memory has to be allocated from self.mem! Otherwise
        # the lexemes will be freed at the end of the function.
        lexemes = <LexemeC*>self.mem.alloc(self.length, sizeof(LexemeC))

        # Temporary buffers
        cdef size_t addr
        cdef hash_t key
        cdef int i = 0
        for key, addr in self._by_hash.items():
            lexeme = <LexemeC*>addr
            lexemes[i] = lexeme[0]
            i += 1
        i += 1
        assert i == self.length, 'Wrote %d lexemes, but Vocab is length %d' % (i, self.length)

        cdef CFile fp = CFile(bytes_loc, 'wb')
        # Write length
        fp.write_from(&self.length, 1, sizeof(self.length))
 
        # Allocate memory for a buffers of flags_t, attr_t and float
        flags_buffer = <flags_t*>tmp_mem.alloc(self.length, sizeof(flags_t))
        attr_buffer = <attr_t*>tmp_mem.alloc(self.length, sizeof(attr_t))
        float_buffer = <float*>tmp_mem.alloc(self.length, sizeof(float))

        # Now fill arrays of data from he lexemes, and write them out.
        # We have to iterate over the lexemes several times here --- but it's
        # much better to do it this way than to do smaller writes.

        # Write out LexemeC.flags
        for i in range(self.length):
            flags_buffer[i] = lexemes[i].flags
        fp.write_from(flags_buffer, self.length, sizeof(flags_t))
        # Write out LexemeC.id
        for i in range(self.length):
            attr_buffer[i] = lexemes[i].id
        fp.write_from(attr_buffer, self.length, sizeof(attr_t))
        # Write out LexemeC.length
        for i in range(self.length):
            attr_buffer[i] = lexemes[i].length
        fp.write_from(attr_buffer, self.length, sizeof(attr_t))
        # Write out LexemeC.orth
        for i in range(self.length):
            attr_buffer[i] = lexemes[i].orth
        fp.write_from(attr_buffer, self.length, sizeof(attr_t))
        # Write out LexemeC.lower
        for i in range(self.length):
            attr_buffer[i] = lexemes[i].lower
        fp.write_from(attr_buffer, self.length, sizeof(attr_t))
        # Write out LexemeC.norm
        for i in range(self.length):
            attr_buffer[i] = lexemes[i].norm
        fp.write_from(attr_buffer, self.length, sizeof(attr_t))
        # Write out LexemeC.shape
        for i in range(self.length):
            attr_buffer[i] = lexemes[i].shape
        fp.write_from(attr_buffer, self.length, sizeof(attr_t))
        # Write out LexemeC.prefix
        for i in range(self.length):
            attr_buffer[i] = lexemes[i].prefix
        fp.write_from(attr_buffer, self.length, sizeof(attr_t))
        # Write out LexemeC.suffix
        for i in range(self.length):
            attr_buffer[i] = lexemes[i].suffix
        fp.write_from(attr_buffer, self.length, sizeof(attr_t))
        # Write out LexemeC.cluster
        for i in range(self.length):
            attr_buffer[i] = lexemes[i].cluster
        fp.write_from(attr_buffer, self.length, sizeof(attr_t))
        # Write out LexemeC.prob
        for i in range(self.length):
            float_buffer[i] = lexemes[i].prob
        fp.write_from(float_buffer, self.length, sizeof(attr_t))
        # Write out LexemeC.sentiment
        for i in range(self.length):
            float_buffer[i] = lexemes[i].sentiment
        fp.write_from(float_buffer, self.length, sizeof(attr_t))
        # Write out LexemeC.l2_norm
        for i in range(self.length):
            float_buffer[i] = lexemes[i].l2_norm
        fp.write_from(float_buffer, self.length, sizeof(attr_t))
        fp.close()

    def load_lexemes(self, loc):
        if not path.exists(loc):
            raise IOError('LexemeCs file not found at %s' % loc)
        cdef LexemeC* lexeme

        check_sizes()

        fp = CFile(loc, 'rb')
        # Read length
        fp.read_into(&self.length, 1, sizeof(self.length))
        # Allocate memory for the lexemes
        # Note that this memory has to be allocated from self.mem! Otherwise
        # the lexemes will be freed at the end of the function.
        lexemes = <LexemeC*>self.mem.alloc(self.length, sizeof(LexemeC))

        # Temporary buffers
        cdef Pool tmp_mem = Pool()
        # Allocate memory for a buffer of flags_t
        flags_buffer = <flags_t*>tmp_mem.alloc(self.length, sizeof(flags_t))
        # Allocate memory for a buffer of attr_t
        attr_buffer = <attr_t*>tmp_mem.alloc(self.length, sizeof(attr_t))
        # Allocate memory for a buffer of float
        float_buffer = <float*>tmp_mem.alloc(self.length, sizeof(float))

        # Now read in arrays of data, and allocate them to the lexemes.
        # We have to iterate over the vocab several times here --- but it's
        # much better to do it this way than to do smaller reads.

        # Read in LexemeC.flags
        fp.read_into(flags_buffer, self.length, sizeof(flags_t))
        for i in range(self.length):
            lexemes[i].flags = flags_buffer[i]
        # Read in LexemeC.id
        fp.read_into(attr_buffer, self.length, sizeof(attr_t))
        for i in range(self.length):
            lexemes[i].id = attr_buffer[i]
        # Read in LexemeC.length
        fp.read_into(attr_buffer, self.length, sizeof(attr_t))
        for i in range(self.length):
            lexemes[i].length = attr_buffer[i]
        # Read in LexemeC.orth
        fp.read_into(attr_buffer, self.length, sizeof(attr_t))
        for i in range(self.length):
            lexemes[i].orth = attr_buffer[i]
        # Read in LexemeC.lower
        fp.read_into(attr_buffer, self.length, sizeof(attr_t))
        for i in range(self.length):
            lexemes[i].lower = attr_buffer[i]
        # Read in LexemeC.norm
        fp.read_into(attr_buffer, self.length, sizeof(attr_t))
        for i in range(self.length):
            lexemes[i].norm = attr_buffer[i]
        # Read in LexemeC.shape
        fp.read_into(attr_buffer, self.length, sizeof(attr_t))
        for i in range(self.length):
            lexemes[i].shape = attr_buffer[i]
        # Read in LexemeC.prefix
        fp.read_into(attr_buffer, self.length, sizeof(attr_t))
        for i in range(self.length):
            lexemes[i].prefix = attr_buffer[i]
        # Read in LexemeC.suffix
        fp.read_into(attr_buffer, self.length, sizeof(attr_t))
        for i in range(self.length):
            lexemes[i].suffix = attr_buffer[i]
        # Read in LexemeC.cluster
        fp.read_into(attr_buffer, self.length, sizeof(attr_t))
        for i in range(self.length):
            lexemes[i].cluster = attr_buffer[i]
        # Read in LexemeC.prob
        fp.read_into(float_buffer, self.length, sizeof(float))
        for i in range(self.length):
            lexemes[i].prob = float_buffer[i]
        # Read in LexemeC.sentiment
        fp.read_into(float_buffer, self.length, sizeof(float))
        for i in range(self.length):
            lexemes[i].sentiment = float_buffer[i]
        # Read in LexemeC.l2_norm
        fp.read_into(float_buffer, self.length, sizeof(float))
        for i in range(self.length):
            lexemes[i].l2_norm = float_buffer[i]
        # Set vector to EMPTY_VEC
        for i in range(self.length):
            lexemes[i].vector = EMPTY_VEC
        # Insert lexemes into hash table
        cdef hash_t key
        cdef unicode py_str
        cdef attr_t orth
        for i in range(self.length):
            py_str = self.strings[lexemes[i].orth]
            key = hash_string(py_str)
            self._by_hash.set(key, &lexemes[i])
            self._by_orth.set(lexemes[i].orth, &lexemes[i])
        fp.close()

    def dump_vectors(self, out_loc):
        cdef int32_t vec_len = self.vectors_length
        cdef int32_t word_len
        cdef bytes word_str
        cdef char* chars
        
        cdef Lexeme lexeme
        cdef CFile out_file = CFile(out_loc, 'wb')
        for lexeme in self:
            word_str = lexeme.orth_.encode('utf8')
            vec = lexeme.c.vector
            word_len = len(word_str)

            out_file.write_from(&word_len, 1, sizeof(word_len))
            out_file.write_from(&vec_len, 1, sizeof(vec_len))

            chars = <char*>word_str
            out_file.write_from(chars, word_len, sizeof(char))
            out_file.write_from(vec, vec_len, sizeof(float))
        out_file.close()

    def load_vectors(self, file_):
        cdef LexemeC* lexeme
        cdef attr_t orth
        cdef int32_t vec_len = -1
        for line_num, line in enumerate(file_):
            pieces = line.split()
            word_str = pieces.pop(0)
            if vec_len == -1:
                vec_len = len(pieces)
            elif vec_len != len(pieces):
                raise VectorReadError.mismatched_sizes(file_, line_num,
                                                        vec_len, len(pieces))
            orth = self.strings[word_str]
            lexeme = <LexemeC*><void*>self.get_by_orth(self.mem, orth)
            lexeme.vector = <float*>self.mem.alloc(self.vectors_length, sizeof(float))

            for i, val_str in enumerate(pieces):
                lexeme.vector[i] = float(val_str)
        return vec_len

    def load_vectors_from_bin_loc(self, loc):
        cdef CFile file_ = CFile(loc, b'rb')
        cdef int32_t word_len
        cdef int32_t vec_len = 0
        cdef int32_t prev_vec_len = 0
        cdef float* vec
        cdef Address mem
        cdef attr_t string_id
        cdef bytes py_word
        cdef vector[float*] vectors
        cdef int line_num = 0
        cdef Pool tmp_mem = Pool()
        while True:
            try:
                file_.read_into(&word_len, sizeof(word_len), 1)
            except IOError:
                break
            file_.read_into(&vec_len, sizeof(vec_len), 1)
            if prev_vec_len != 0 and vec_len != prev_vec_len:
                raise VectorReadError.mismatched_sizes(loc, line_num,
                                                       vec_len, prev_vec_len)
            if 0 >= vec_len >= MAX_VEC_SIZE:
                raise VectorReadError.bad_size(loc, vec_len)

            chars = <char*>file_.alloc_read(tmp_mem, word_len, sizeof(char))
            vec = <float*>file_.alloc_read(self.mem, vec_len, sizeof(float))

            string_id = self.strings[chars[:word_len]]
            while string_id >= vectors.size():
                vectors.push_back(EMPTY_VEC)
            assert vec != NULL
            vectors[string_id] = vec
            line_num += 1
        cdef LexemeC* lex
        cdef size_t lex_addr
        cdef int i
        for orth, lex_addr in self._by_orth.items():
            lex = <LexemeC*>lex_addr
            if lex.lower < vectors.size():
                lex.vector = vectors[lex.lower]
                for i in range(vec_len):
                    lex.l2_norm += (lex.vector[i] * lex.vector[i])
                lex.l2_norm = math.sqrt(lex.l2_norm)
            else:
                lex.vector = EMPTY_VEC
        return vec_len


def unpickle_vocab(strings_loc, lex_loc, vec_loc, morphology, get_lex_attr,
                   serializer_freqs, data_dir):
    cdef Vocab vocab = Vocab()

    vocab.get_lex_attr = get_lex_attr
    vocab.morphology = morphology
    vocab.strings = morphology.strings
    vocab.data_dir = data_dir
    vocab.serializer_freqs = serializer_freqs

    with io.open(strings_loc, 'r', encoding='utf8') as file_:
        vocab.strings.load(file_)
    vocab.load_lexemes(lex_loc)
    if vec_loc is not None:
        vocab.vectors_length = vocab.load_vectors_from_bin_loc(vec_loc)
    return vocab
 

copy_reg.constructor(unpickle_vocab)


def write_binary_vectors(in_loc, out_loc):
    cdef CFile out_file = CFile(out_loc, 'wb')
    cdef Address mem
    cdef int32_t word_len
    cdef int32_t vec_len
    cdef char* chars
    with bz2.BZ2File(in_loc, 'r') as file_:
        for line in file_:
            pieces = line.split()
            word = pieces.pop(0)
            mem = Address(len(pieces), sizeof(float))
            vec = <float*>mem.ptr
            for i, val_str in enumerate(pieces):
                vec[i] = float(val_str)

            word_len = len(word)
            vec_len = len(pieces)

            out_file.write_from(&word_len, 1, sizeof(word_len))
            out_file.write_from(&vec_len, 1, sizeof(vec_len))

            chars = <char*>word
            out_file.write_from(chars, len(word), sizeof(char))
            out_file.write_from(vec, vec_len, sizeof(float))


class LookupError(Exception):
    @classmethod
    def mismatched_strings(cls, id_, id_string, original_string):
        return cls(
            "Error fetching a Lexeme from the Vocab. When looking up a string, "
            "the lexeme returned had an orth ID that did not match the query string. "
            "This means that the cached lexeme structs are mismatched to the "
            "string encoding table. The mismatched:\n"
            "Query string: {query}\n"
            "Orth cached: {orth_str}\n"
            "ID of orth: {orth_id}".format(
                query=repr(original_string), orth_str=repr(id_string), orth_id=id_)
        )


class VectorReadError(Exception):
    @classmethod
    def mismatched_sizes(cls, loc, line_num, prev_size, curr_size):
        return cls(
            "Error reading word vectors from %s on line %d.\n"
            "All vectors must be the same size.\n"
            "Prev size: %d\n"
            "Curr size: %d" % (loc, line_num, prev_size, curr_size))

    @classmethod
    def bad_size(cls, loc, size):
        return cls(
            "Error reading word vectors from %s.\n"
            "Vector size: %d\n"
            "Max size: %d\n"
            "Min size: 1\n" % (loc, size, MAX_VEC_SIZE))


def check_sizes():
    cdef LexemeC lexeme
    # Check our assumptions about how everything is sized
    assert sizeof(flags_t) == 8
    assert sizeof(attr_t) == 4
    assert sizeof(float) == 4
    assert sizeof(lexeme.flags) == sizeof(flags_t)
    assert sizeof(lexeme.id) == sizeof(attr_t)
    assert sizeof(lexeme.length) == sizeof(attr_t)
    assert sizeof(lexeme.orth) == sizeof(attr_t)
    assert sizeof(lexeme.lower) == sizeof(attr_t)
    assert sizeof(lexeme.norm) == sizeof(attr_t)
    assert sizeof(lexeme.shape) == sizeof(attr_t)
    assert sizeof(lexeme.suffix) == sizeof(attr_t)
    assert sizeof(lexeme.cluster) == sizeof(attr_t)
    assert sizeof(lexeme.prob) == sizeof(float)
    assert sizeof(lexeme.sentiment) == sizeof(float)
    assert sizeof(lexeme.l2_norm) == sizeof(float)
