from libc.string cimport memcpy, memset
from libc.stdint cimport uint32_t
from ..vocab cimport EMPTY_LEXEME
from ..structs cimport EntityC
from ..lexeme cimport Lexeme
from ..symbols cimport punct
from ..attrs cimport IS_SPACE


cdef class StateClass:
    def __init__(self, int length):
        cdef Pool mem = Pool()
        cdef int PADDING = 5
        self.mem = mem
        self.c._buffer = <int*>mem.alloc(length + (PADDING * 2), sizeof(int))
        self.c._stack = <int*>mem.alloc(length + (PADDING * 2), sizeof(int))
        self.c.shifted = <bint*>mem.alloc(length + (PADDING * 2), sizeof(bint))
        self.c.sent = <TokenC*>mem.alloc(length + (PADDING * 2), sizeof(TokenC))
        self.c.ents = <EntityC*>mem.alloc(length + (PADDING * 2), sizeof(EntityC))
        cdef int i
        for i in range(length + (PADDING * 2)):
            self.c.ents[i].end = -1
            self.c.sent[i].l_edge = i
            self.c.sent[i].r_edge = i
        for i in range(length, length + (PADDING * 2)):
            self.c.sent[i].lex = &EMPTY_LEXEME
        self.c.sent += PADDING
        self.c.ents += PADDING
        self.c._buffer += PADDING
        self.c._stack += PADDING
        self.c.shifted += PADDING
        self.c.length = length
        self.c._break = -1
        self.c._s_i = 0
        self.c._b_i = 0
        self.c._e_i = 0
        for i in range(length):
            self.c._buffer[i] = i
        self.c.empty_token.lex = &EMPTY_LEXEME

    @property
    def stack(self):
        return {self.S(i) for i in range(self.c._s_i)}

    @property
    def queue(self):
        return {self.B(i) for i in range(self.c._b_i)}

    cdef int E(self, int i) nogil:
        if self.c._e_i <= 0 or self.c._e_i >= self.c.length:
            return 0
        if i < 0 or i >= self.c._e_i:
            return 0
        return self.c.ents[self.c._e_i - (i+1)].start

    cdef int L(self, int i, int idx) nogil:
        if idx < 1:
            return -1
        if i < 0 or i >= self.c.length:
            return -1
        cdef const TokenC* target = &self.c.sent[i]
        if target.l_kids < idx:
            return -1
        cdef const TokenC* ptr = &self.c.sent[target.l_edge]

        while ptr < target:
            # If this head is still to the right of us, we can skip to it
            # No token that's between this token and this head could be our
            # child.
            if (ptr.head >= 1) and (ptr + ptr.head) < target:
                ptr += ptr.head

            elif ptr + ptr.head == target:
                idx -= 1
                if idx == 0:
                    return ptr - self.c.sent
                ptr += 1
            else:
                ptr += 1
        return -1

    cdef int R(self, int i, int idx) nogil:
        if idx < 1:
            return -1
        if i < 0 or i >= self.c.length:
            return -1
        cdef const TokenC* target = &self.c.sent[i]
        if target.r_kids < idx:
            return -1
        cdef const TokenC* ptr = &self.c.sent[target.r_edge]
        while ptr > target:
            # If this head is still to the right of us, we can skip to it
            # No token that's between this token and this head could be our
            # child.
            if (ptr.head < 0) and ((ptr + ptr.head) > target):
                ptr += ptr.head
            elif ptr + ptr.head == target:
                idx -= 1
                if idx == 0:
                    return ptr - self.c.sent
                ptr -= 1
            else:
                ptr -= 1
        return -1

    cdef void push(self) nogil:
        if self.B(0) != -1:
            self.c._stack[self.c._s_i] = self.B(0)
        self.c._s_i += 1
        self.c._b_i += 1
        if self.c._b_i > self.c._break:
            self.c._break = -1

    cdef void pop(self) nogil:
        if self.c._s_i >= 1:
            self.c._s_i -= 1

    cdef void unshift(self) nogil:
        self.c._b_i -= 1
        self.c._buffer[self.c._b_i] = self.S(0)
        self.c._s_i -= 1
        self.c.shifted[self.B(0)] = True

    cdef void fast_forward(self) nogil:
        while self.buffer_length() == 0 \
        or self.stack_depth() == 0 \
        or Lexeme.c_check_flag(self.S_(0).lex, IS_SPACE):
            if self.buffer_length() == 1 and self.stack_depth() == 0:
                self.push()
                self.pop()
            elif self.buffer_length() == 0 and self.stack_depth() == 1:
                self.pop()
            elif self.buffer_length() == 0 and self.stack_depth() >= 2:
                if self.has_head(self.S(0)):
                    self.pop()
                else:
                    self.unshift()
            elif (self.c.length - self.c._b_i) >= 1 and self.stack_depth() == 0:
                self.push()
            elif Lexeme.c_check_flag(self.S_(0).lex, IS_SPACE):
                self.add_arc(self.B(0), self.S(0), 0)
                self.pop()
            else:
                break

    cdef void add_arc(self, int head, int child, int label) nogil:
        if self.has_head(child):
            self.del_arc(self.H(child), child)

        cdef int dist = head - child
        self.c.sent[child].head = dist
        self.c.sent[child].dep = label
        cdef int i
        if child > head:
            self.c.sent[head].r_kids += 1
            # Some transition systems can have a word in the buffer have a
            # rightward child, e.g. from Unshift.
            self.c.sent[head].r_edge = self.c.sent[child].r_edge
            i = 0
            while self.has_head(head) and i < self.c.length:
                head = self.H(head)
                self.c.sent[head].r_edge = self.c.sent[child].r_edge
                i += 1 # Guard against infinite loops
        else:
            self.c.sent[head].l_kids += 1
            self.c.sent[head].l_edge = self.c.sent[child].l_edge

    cdef void del_arc(self, int h_i, int c_i) nogil:
        cdef int dist = h_i - c_i
        cdef TokenC* h = &self.c.sent[h_i]
        if c_i > h_i:
            h.r_edge = self.R_(h_i, 2).r_edge if h.r_kids >= 2 else h_i
            h.r_kids -= 1
        else:
            h.l_edge = self.L_(h_i, 2).l_edge if h.l_kids >= 2 else h_i
            h.l_kids -= 1

    cdef void open_ent(self, int label) nogil:
        self.c.ents[self.c._e_i].start = self.B(0)
        self.c.ents[self.c._e_i].label = label
        self.c.ents[self.c._e_i].end = -1
        self.c._e_i += 1

    cdef void close_ent(self) nogil:
        # Note that we don't decrement _e_i here! We want to maintain all
        # entities, not over-write them...
        self.c.ents[self.c._e_i-1].end = self.B(0)+1
        self.c.sent[self.B(0)].ent_iob = 1

    cdef void set_ent_tag(self, int i, int ent_iob, int ent_type) nogil:
        if 0 <= i < self.c.length:
            self.c.sent[i].ent_iob = ent_iob
            self.c.sent[i].ent_type = ent_type

    cdef void set_break(self, int _) nogil:
        if 0 <= self.B(0) < self.c.length: 
            self.c.sent[self.B(0)].sent_start = True
            self.c._break = self.c._b_i

    cdef void clone(self, StateClass src) nogil:
        memcpy(self.c.sent, src.c.sent, self.c.length * sizeof(TokenC))
        memcpy(self.c._stack, src.c._stack, self.c.length * sizeof(int))
        memcpy(self.c._buffer, src.c._buffer, self.c.length * sizeof(int))
        memcpy(self.c.ents, src.c.ents, self.c.length * sizeof(EntityC))
        self.c._b_i = src.c._b_i
        self.c._s_i = src.c._s_i
        self.c._e_i = src.c._e_i
        self.c._break = src.c._break

    def print_state(self, words):
        words = list(words) + ['_']
        top = words[self.S(0)] + '_%d' % self.S_(0).head
        second = words[self.S(1)] + '_%d' % self.S_(1).head
        third = words[self.S(2)] + '_%d' % self.S_(2).head
        n0 = words[self.B(0)] 
        n1 = words[self.B(1)] 
        return ' '.join((third, second, top, '|', n0, n1))
