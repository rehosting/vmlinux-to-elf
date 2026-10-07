#!/usr/bin/env python3
# -*- encoding: Utf-8 -*-
"""
Exported symbols of a kernel built without CONFIG_KALLSYMS, recovered
from its export tables: __ksymtab / __ksymtab_gpl (one entry per
EXPORT_SYMBOL) and __ksymtab_strings (the exported names).

Used by ElfSymbolizer when kallsyms is absent and the caller asked for
it (ksymtab_fallback=True / --ksymtab-fallback), or always
(ksymtab_only=True / --ksymtab-only).

Entry layouts (include/linux/export.h, struct kernel_symbol):

* absolute, all kernels up to 4.18 and later ones without
  HAVE_ARCH_PREL32_RELOCATIONS: { unsigned long value; const char
  *name; } and, since 5.4, a third `const char *namespace` word.
* PREL32 (4.19+, arm64/x86/powerpc/riscv/...): { int value_offset;
  int name_offset; } and, since 5.4, a third `int namespace_offset`;
  each field holds the target address minus the field's own address.

The tables are found by their structure, without any symbol:

1. string tables: runs of NUL-separated C identifiers (empty namespace
   strings and merged string tails allowed);
2. PREL32 entries: every 4-aligned int32 of the image is tried as a
   name_offset; an entry run is a run of fields at a fixed stride (8 or
   12 bytes) that all land on an identifier of the same string table
   (and, with 12 bytes, whose namespace field lands in the table too).
   Both byte orders are tried;
3. absolute entries: the base address B is voted for by the image's
   words x and the table's string starts s with x - B = s (B assumed
   page aligned, so only starts with x's low 12 bits vote); entries are
   then the runs of words, at a stride of 2 or 3 words, whose name word
   minus B lands on an identifier of the table.

The longest set of runs wins. Every run of at least MIN_RUN entries
pointing into that table is kept (__ksymtab, __ksymtab_gpl, and the
older __ksymtab_unused* / __ksymtab_gpl_future).

Base address (the virtual address of the first byte of the image), in
this order: the caller's base address; for absolute entries the vote
above; for PREL32 entries, which do not depend on it, two votes by the
image's pointer-sized words: a word x at offset p votes for B = x - p
(a self pointer, e.g. an empty list head), and each distinct word x for
B = x - o for every exported symbol at offset o with the same low 12
bits (a pointer to an exported symbol); both must agree, or one must be
clear and the other silent (vote_prel32_base). An arm64 image with a relocation table is relocated
first (KallsymsFinder.find_elf64_rela / apply_elf64_rela), since its
pointers are only filled in at boot. A base that no clear majority of
the votes supports is an error: pass --base-address.

An ELF input (a stripped vmlinux) is flattened over its main cluster of
allocated sections, whose addresses give the base.

Sections of a raw image (core layout of include/asm-generic/
vmlinux.lds.h: text, then RO_DATA holding __ksymtab* and
__ksymtab_strings, then init and data), as far as they can be told
without linker symbols:

  .text    [image start, rodata start)   AX
  .rodata  [rodata start, end of __ksymtab_strings)  A
  .data    [end of __ksymtab_strings, image end)  AW  (init text and
           data and .data are not told apart)

where rodata start is the kernel banner ("Linux version ...", an early
.rodata object) when it lies below the export tables, else the start of
the export tables. Exported symbols in .text are typed as functions,
the others as data.
"""

import logging
from array import array
from collections import Counter, defaultdict
from re import compile as re_compile
from struct import unpack_from
from typing import List, Optional, Tuple

from vmlinux_to_elf.core.architecture_detecter import (
    ArchitectureGuessError,
    ArchitectureName,
)
from vmlinux_to_elf.core.kallsyms import (
    KallsymsFinder,
    KallsymsSymbol,
    KallsymsSymbolType,
)
from vmlinux_to_elf.utils.elf import SH_FLAGS, ElfNoBits, ElfProgbits

# A run of at least this many NUL-separated identifiers may be
# __ksymtab_strings
MIN_STRINGS = 32
# Entry runs shorter than this are ignored
MIN_RUN = 16
# Share of a run's names that must start a string
MIN_STARTS = 0.9
# What the export tables found must look like: this many entries, naming
# this share of the strings of their table, and (absolute entries) with
# this share of the values in or just after the image
MIN_ENTRIES = 64
MIN_NAMED_STRINGS = 0.5
MIN_VALUES_IN_IMAGE = 0.9
# String tables tried, largest first
MAX_STRING_TABLES = 4
# __ksymtab* lie below __ksymtab_strings in RO_DATA; search this far below
# first, then the whole image
NEAR_WINDOW = 0x400000
# Base address votes: the winner needs this many votes, and this many
# times the runner-up's
MIN_BASE_VOTES = 8
BASE_VOTE_RATIO = 3

PAGE_MASK = 0xFFF

STRINGS_RE = re_compile(rb'(?:[A-Za-z_][\w.:\-]*\x00+){%d,}' % MIN_STRINGS)
IDENTIFIER_RE = re_compile(rb'[A-Za-z_]\w*\x00')
NAMESPACE_RE = re_compile(rb'[\w.:\-]*\x00')
BANNER_RE = re_compile(rb'Linux version \d+\.[\d.]*\d[ -~]+')


class KsymtabNotFoundException(ValueError):
    pass


class StringTable:
    """A candidate __ksymtab_strings: file offsets [start, end)."""

    def __init__(self, image: bytes, start: int, end: int):
        self.image = image
        self.start = start
        self.end = end
        self._starts = None

    def name_at(self, offset: int) -> Optional[str]:
        """The identifier at `offset`, if it lies inside the table."""
        if not self.start <= offset < self.end:
            return None
        match = IDENTIFIER_RE.match(self.image, offset)
        if not match or match.end() > self.end:
            return None
        return match.group(0)[:-1].decode('ascii')

    def is_start(self, offset: int) -> bool:
        """
        A string starts at `offset` (not inside another one). In the
        first string of the table any offset counts: the table found may
        begin with a few bytes of the entries before it that read as
        identifier characters.
        """
        return (
            self.image[offset - 1] == 0
            or self.image.find(b'\0', self.start, offset) == -1
        )

    def holds_namespace(self, offset: int) -> bool:
        """A namespace string ("" included) starts at `offset`."""
        if not self.start <= offset < self.end:
            return False
        match = NAMESPACE_RE.match(self.image, offset)
        return bool(match) and match.end() <= self.end

    def string_starts(self) -> List[int]:
        if self._starts is None:
            starts = []
            previous = 0
            for offset in range(self.start, self.end):
                byte = self.image[offset]
                if byte and not previous:
                    starts.append(offset)
                previous = byte
            self._starts = starts
        return self._starts


class EntryRun:
    """Consecutive export entries: `count` entries of `stride` bytes."""

    def __init__(self, start: int, stride: int, count: int):
        self.start = start
        self.stride = stride
        self.count = count

    @property
    def end(self) -> int:
        return self.start + self.stride * self.count

    def __repr__(self):
        return 'EntryRun(%#x, stride=%d, count=%d)' % (
            self.start,
            self.stride,
            self.count,
        )


def find_string_tables(image: bytes) -> List[StringTable]:
    tables = []
    for match in STRINGS_RE.finditer(image):
        # up to the NUL ending the last string, not the padding after it
        end = match.end()
        while image[end - 2] == 0:
            end -= 1
        tables.append(StringTable(image, match.start(), end))
    tables.sort(key=lambda table: table.end - table.start, reverse=True)
    return tables[:MAX_STRING_TABLES]


def collect_runs(
    fields: set, starts: set, stride: int, field_offset: int
) -> List[EntryRun]:
    """
    Runs of entries whose name field at `field_offset` is in `fields`
    (field file offsets, naming an identifier), `stride` bytes apart,
    of which nearly all (MIN_STARTS) name a whole string (`starts`):
    a name merged into the tail of another string is rare, while a
    field that only happens to land inside the table mostly lands in
    the middle of a string.
    """
    runs = []
    for field in sorted(fields):
        if field - stride in fields:
            continue
        run = []
        position = field
        while position in fields:
            run.append(position)
            position += stride
        # a run neither starts nor ends on a name inside another string
        while run and run[0] not in starts:
            run.pop(0)
        while run and run[-1] not in starts:
            run.pop()
        strong = sum(position in starts for position in run)
        if len(run) >= MIN_RUN and strong >= MIN_STARTS * len(run):
            runs.append(EntryRun(run[0] - field_offset, stride, len(run)))
    return runs


def flatten_elf(contents: bytes) -> Tuple[bytes, int]:
    """
    (image, base): the allocated sections of an ELF file laid out at
    their addresses, over the cluster of sections (gaps below 16 MiB)
    holding the most bytes, and that cluster's start address. Reads
    the section headers only (a full vmlinux symtab is large).
    """
    is_64_bits = contents[4] == 2
    order = '>' if contents[5] == 2 else '<'
    if is_64_bits:
        (shoff,) = unpack_from(order + 'Q', contents, 0x28)
        shentsize, shnum = unpack_from(order + 'HH', contents, 0x3A)
        header = order + 'IIQQQQIIQQ'
    else:
        (shoff,) = unpack_from(order + 'I', contents, 0x20)
        shentsize, shnum = unpack_from(order + 'HH', contents, 0x2E)
        header = order + 'IIIIIIIIII'
    SHT_NOBITS = 8
    sections = []
    for index in range(shnum):
        _, sh_type, sh_flags, sh_addr, sh_offset, sh_size = unpack_from(
            header, contents, shoff + index * shentsize
        )[:6]
        if (
            sh_flags & SH_FLAGS.SHF_ALLOC
            and sh_addr
            and sh_size
            and sh_type != SHT_NOBITS
        ):
            sections.append((sh_addr, sh_offset, sh_size))
    if not sections:
        raise KsymtabNotFoundException('The ELF input has no loaded sections')
    sections.sort()
    clusters = [[sections[0]]]
    for section in sections[1:]:
        address, _, size = clusters[-1][-1]
        if section[0] - (address + size) < 0x1000000:
            clusters[-1].append(section)
        else:
            clusters.append([section])
    cluster = max(clusters, key=lambda sections: sum(s[2] for s in sections))
    base = cluster[0][0]
    image = bytearray(max(a + size for a, _, size in cluster) - base)
    for address, offset, size in cluster:
        image[address - base : address - base + size] = contents[
            offset : offset + size
        ]
    return bytes(image), base


class KsymtabFinder(KallsymsFinder):
    """
    Same interface as KallsymsFinder (symbols, name_to_symbol,
    elf_machine, is_big_endian, is_64_bits, kernel_text_candidate,
    elf64_rela*) for ElfSymbolizer, plus:

    layout: [(name, start address, end address, flags)] for a raw
        image, None for an ELF input
    entry_point: the image start address (None for an ELF input)
    ksymtab: what was found (summary() gives it as JSON-able values)
    """

    def __init__(
        self,
        kernel_img: bytes,
        bit_size: int = None,
        explicit_base_address: Optional[int] = None,
    ):
        self.kernel_img = kernel_img
        self.explicit_base_address = explicit_base_address
        self.override_relative_base = False
        if bit_size:
            if bit_size not in (64, 32):
                raise ArchitectureGuessError(
                    'Please specify a register bit size of either 32 or 64 bits'
                )
            self.is_64_bits = bit_size == 64

        self.ksymtab = {}
        self.layout = None
        self.elf_base = None

        # Bounds of the kallsyms table, which apply_elf64_rela must not
        # patch: there is none
        self.kallsyms_num_syms__offset = 0
        self.kallsyms_token_index_end__offset = 0
        self.uncompressed_kallsyms = False

        if self.kernel_img.startswith(b'\x7fELF'):
            # (arm64: apply its R_AARCH64_RELATIVE relocations first)
            self.preprocess_elf_rela_table()
            self.kernel_img, self.elf_base = flatten_elf(self.kernel_img)
            logging.info(
                '[+] ELF input: searching its sections laid out from %#x'
                % self.elf_base
            )
            self.is_relocated = True
        else:
            self.preprocess_uimage_header()

        try:
            self.find_linux_kernel_version()
        except ValueError:
            self.version_string = self.version_number = None
            logging.warning('[!] No kernel version string found')

        try:
            self.guess_architecture()
        except ArchitectureGuessError:
            logging.warning(
                '[!] Architecture not guessed; taking the byte order and '
                + 'word size from the export tables'
            )

        if (
            self.is_64_bits
            and not self.is_relocated
            and self.architecture == ArchitectureName.aarch64
        ):
            # Relocatable arm64 Image: pointers (absolute __ksymtab entries
            # included) are only filled in by these relocations
            self.find_elf64_rela()
            self.apply_elf64_rela()

        self.find_ksymtab()
        self.find_base()
        self.make_symbols()
        self.entry_point = None
        if self.elf_base is None:
            self.find_layout()
            self.entry_point = self.kernel_text_candidate

    # -- export tables -------------------------------------------------------

    def find_ksymtab(self):
        tables = find_string_tables(self.kernel_img)
        if not tables:
            raise KsymtabNotFoundException(
                'No table of exported symbol names (__ksymtab_strings) found'
            )
        best = None
        for table in tables:
            for search in (
                self.find_prel32_entries,
                self.find_absolute_entries,
            ):
                found = search(table)
                if found and (best is None or found['count'] > best['count']):
                    best = found
                if self._enough(best):
                    break
            if self._enough(best):
                break
        if best is None:
            raise KsymtabNotFoundException(
                'No __ksymtab entries point into the candidate '
                + '__ksymtab_strings tables (%s)'
                % ', '.join(
                    '%#x-%#x' % (table.start, table.end) for table in tables
                )
            )
        self.check_plausible(best)
        self.ksymtab = best
        self.is_big_endian = best['big_endian']
        if best['kind'] == 'absolute':
            self.is_64_bits = best['word'] == 8
        table = best['table']
        logging.info(
            '[+] Found %d __ksymtab entries (%s, %d bytes, %s-endian) in '
            % (
                best['count'],
                best['kind'],
                best['stride'],
                'big' if best['big_endian'] else 'little',
            )
            + ', '.join(
                '%#x-%#x' % (run.start, run.end) for run in best['runs']
            )
            + ' naming %#x-%#x (__ksymtab_strings)' % (table.start, table.end)
        )

    def check_plausible(self, found):
        """
        Refuse a find that is not a kernel's export table: too few
        entries, naming too few of the table's strings, or (absolute)
        values mostly outside the image for the base they imply. A
        relocatable kernel whose pointers are only filled in at boot
        (other than arm64, whose relocations are applied first) has
        zeros where its absolute entries should be.
        """
        table = found['table']
        names = {name for name, _ in self._entries(found)}
        strings = len(table.string_starts())
        problem = None
        if found['count'] < MIN_ENTRIES:
            problem = 'only %d entries' % found['count']
        elif len(names) < MIN_NAMED_STRINGS * strings:
            problem = 'they name %d of the %d strings of the table' % (
                len(names),
                strings,
            )
        elif found['kind'] == 'absolute':
            low = found['base']
            high = low + len(self.kernel_img) + 0x4000000  # .bss
            inside = sum(
                low <= value < high for _, value in self._entries(found)
            )
            if inside < MIN_VALUES_IN_IMAGE * found['count']:
                problem = '%d of %d values lie outside the image at %#x' % (
                    found['count'] - inside,
                    found['count'],
                    low,
                )
        if problem:
            raise KsymtabNotFoundException(
                'No plausible __ksymtab: the best find (%s entries naming '
                % found['kind']
                + '%#x-%#x) has %s; a relocatable kernel with its '
                % (table.start, table.end, problem)
                + 'pointers unfilled, or no module support?'
            )

    def _search_windows(self, table: StringTable):
        # (the table found may start inside the last entry, when the
        # entry's last bytes read as identifier characters)
        near = (
            max(0, table.start - NEAR_WINDOW),
            min(len(self.kernel_img), table.start + 64),
        )
        return [near, (0, len(self.kernel_img))]

    def _byte_orders(self):
        """The guessed byte order first (a guess, not a constraint)."""
        if self.is_big_endian is None:
            return (False, True)
        return (self.is_big_endian, not self.is_big_endian)

    def _word_sizes(self):
        """The given (or guessed) word size first."""
        if self.is_64_bits is None:
            return (4, 8)
        return (8, 4) if self.is_64_bits else (4, 8)

    @staticmethod
    def _enough(found) -> bool:
        return bool(found) and found['count'] >= 4 * MIN_RUN

    def _words(self, word, big_endian, signed, start, end):
        """(aligned start, array of the `word`-byte integers in
        [start, end) of the image)."""
        values = array(
            {4: 'i', 8: 'q'}[word] if signed else {4: 'I', 8: 'Q'}[word]
        )
        assert values.itemsize == word
        start -= start % word
        end -= (end - start) % word
        values.frombytes(self.kernel_img[start:end])
        if big_endian != (array('H', b'\x00\x01')[0] == 1):
            values.byteswap()
        return start, values

    def find_prel32_entries(self, table: StringTable) -> Optional[dict]:
        best = None
        for window in self._search_windows(table):
            for big_endian in self._byte_orders():
                found = self._prel32_runs(table, window, big_endian)
                if found and (best is None or found['count'] > best['count']):
                    best = found
            if best:
                break  # the window below the strings was enough
        return best

    def _prel32_runs(self, table, window, big_endian) -> Optional[dict]:
        start, values = self._words(4, big_endian, True, *window)
        fields = {
            start + 4 * index
            for index, relative in enumerate(values)
            if table.start <= start + 4 * index + relative < table.end
        }
        # A name field follows a value field, which does not point at a
        # string (as in an array of string references)
        # (and no entry starts inside the string table)
        fields = {
            field
            for field in fields
            if field - 4 not in fields
            and not table.start <= field - 4 < table.end
            and table.name_at(field + self._s32(field, big_endian))
        }
        starts = {
            field
            for field in fields
            if table.is_start(field + self._s32(field, big_endian))
        }
        found = None
        for stride in (12, 8):
            runs = collect_runs(fields, starts, stride, 4)
            if stride == 12:
                runs = [
                    self._namespace_prefix(run, table, big_endian)
                    for run in runs
                ]
                runs = [run for run in runs if run.count >= MIN_RUN]
            count = sum(run.count for run in runs)
            if count and (found is None or count > found['count']):
                found = {
                    'kind': 'prel32',
                    'stride': stride,
                    'word': 4,
                    'big_endian': big_endian,
                    'runs': runs,
                    'count': count,
                    'table': table,
                }
        return found

    def _s32(self, offset: int, big_endian: bool) -> int:
        return unpack_from(
            '>i' if big_endian else '<i', self.kernel_img, offset
        )[0]

    def _namespace_prefix(
        self, run: EntryRun, table: StringTable, big_endian: bool
    ) -> EntryRun:
        """The leading entries of a 12-byte run whose namespace field
        points into the string table."""
        count = 0
        for index in range(run.count):
            field = run.start + index * 12 + 8
            relative = self._s32(field, big_endian)
            # 0: no namespace (5.4 - 5.9)
            if relative and not table.holds_namespace(field + relative):
                break
            count += 1
        return EntryRun(run.start, run.stride, count)

    def find_absolute_entries(self, table: StringTable) -> Optional[dict]:
        starts = table.string_starts()
        # Vote with a sample of the string starts (those at a multiple of
        # `sample`), enough for about 256 votes to the right base
        sample = 1
        while len(starts) // (2 * sample) >= 256:
            sample *= 2
        starts_by_page_offset = defaultdict(list)
        for offset in starts:
            if not offset % sample:
                starts_by_page_offset[offset & PAGE_MASK].append(offset)
        best = None
        for window in self._search_windows(table):
            for word in self._word_sizes():
                for big_endian in self._byte_orders():
                    start, values = self._words(
                        word, big_endian, False, *window
                    )
                    if self.explicit_base_address is not None:
                        candidates = [self.explicit_base_address]
                    else:
                        candidates = self._vote_absolute_base(
                            values, starts_by_page_offset, sample
                        )
                    for base in candidates:
                        found = self._absolute_runs(
                            table, start, values, word, big_endian, base
                        )
                        if found:
                            if best is None or found['count'] > best['count']:
                                best = found
                            break
            if best:
                break  # the window below the strings was enough
        return best

    @staticmethod
    def _vote_absolute_base(
        values, starts_by_page_offset, sample
    ) -> List[int]:
        votes = Counter()
        for value in {value for value in values if not value % sample}:
            for string_start in starts_by_page_offset.get(
                value & PAGE_MASK, ()
            ):
                votes[value - string_start] += 1
        return [
            base
            for base, count in votes.most_common(3)
            if count >= MIN_RUN // 4 and base > 0
        ]

    def _absolute_runs(self, table, start, values, word, big_endian, base):
        fields, starts = set(), set()
        for index, value in enumerate(values):
            offset = value - base
            if table.start <= offset < table.end and table.name_at(offset):
                fields.add(start + word * index)
                if table.is_start(offset):
                    starts.add(start + word * index)
        # A name word follows a value word, which does not point at a
        # string (as in an array of string pointers)
        # (and no entry starts inside the string table)
        fields = {
            field
            for field in fields
            if field - word not in fields
            and not table.start <= field - word < table.end
        }
        best = None
        for words_per_entry in (3, 2):
            stride = word * words_per_entry
            runs = collect_runs(fields, starts, stride, word)
            if words_per_entry == 3:
                runs = [
                    self._absolute_namespace_prefix(
                        run, table, word, big_endian, base
                    )
                    for run in runs
                ]
                runs = [run for run in runs if run.count >= MIN_RUN]
            count = sum(run.count for run in runs)
            if count and (best is None or count > best['count']):
                best = {
                    'kind': 'absolute',
                    'stride': stride,
                    'word': word,
                    'big_endian': big_endian,
                    'runs': runs,
                    'count': count,
                    'table': table,
                    'base': base,
                }
        return best

    def _word(self, offset: int, word: int, big_endian: bool) -> int:
        return unpack_from(
            ('>' if big_endian else '<') + ('Q' if word == 8 else 'I'),
            self.kernel_img,
            offset,
        )[0]

    def _absolute_namespace_prefix(self, run, table, word, big_endian, base):
        count = 0
        for index in range(run.count):
            pointer = self._word(
                run.start + index * run.stride + 2 * word, word, big_endian
            )
            if not (pointer == 0 or table.holds_namespace(pointer - base)):
                break
            count += 1
        return EntryRun(run.start, run.stride, count)

    def entries(self) -> List[Tuple[str, int]]:
        """[(name, value)] in table order: value is a file offset (PREL32)
        or an address (absolute)."""
        return self._entries(self.ksymtab)

    def _entries(self, found) -> List[Tuple[str, int]]:
        table = found['table']
        big_endian = found['big_endian']
        out = []
        for run in found['runs']:
            for index in range(run.count):
                entry = run.start + index * run.stride
                if found['kind'] == 'prel32':
                    value = entry + self._s32(entry, big_endian)
                    name = table.name_at(
                        entry + 4 + self._s32(entry + 4, big_endian)
                    )
                else:
                    word = found['word']
                    value = self._word(entry, word, big_endian)
                    name = table.name_at(
                        self._word(entry + word, word, big_endian)
                        - found['base']
                    )
                out.append((name, value))
        return out

    # -- base address --------------------------------------------------------

    def find_base(self):
        found = self.ksymtab
        if self.elf_base is not None:
            base, how = self.elf_base, 'the ELF section addresses'
        elif self.explicit_base_address is not None:
            base, how = self.explicit_base_address, 'provided'
        elif found['kind'] == 'absolute':
            base, how = found['base'], 'the __ksymtab name pointers'
        else:
            base, votes = self.vote_prel32_base()
            how = 'pointers to exported symbols and self pointers (%s)' % votes
        if found['kind'] == 'absolute' and found['base'] != base:
            raise KsymtabNotFoundException(
                'The __ksymtab name pointers give the base address %#x, '
                % found['base']
                + 'not %#x (%s)' % (base, how)
            )
        self.kernel_text_candidate = base
        self.ksymtab['base_from'] = how
        logging.info('[+] Base address %#x, from %s' % (base, how))

    def vote_prel32_base(self) -> Tuple[int, str]:
        """
        The base address the image's pointer-sized words agree on, by
        two independent votes: self pointers (a word at offset p holding
        B + p) and pointers to exported symbols (a word holding B + o
        for an exported symbol at offset o). The base is accepted when
        both votes put it first, or when one put it first by
        BASE_VOTE_RATIO times the runner-up and the other is not against
        it (no votes, or the same winner). Pointer-sized words of the
        known (else 64-bit, then 32-bit) size; the first size with a
        verdict wins.
        """
        offsets_by_page_offset = defaultdict(list)
        for _, offset in self.entries():
            offsets_by_page_offset[offset & PAGE_MASK].append(offset)
        best = None
        # 64-bit first when unknown: the low halves of 64-bit pointers
        # agree on the low half of the base
        for word in (4, 8) if self.is_64_bits is False else (8, 4):
            _, values = self._words(
                word, self.is_big_endian, False, 0, len(self.kernel_img)
            )
            self_votes = Counter()
            for index, value in enumerate(values):
                if value and not (value - index * word) & PAGE_MASK:
                    self_votes[value - index * word] += 1
            # One vote per distinct value: a repeated instruction pair is
            # not a pointer
            export_votes = Counter()
            for value in set(values):
                for offset in offsets_by_page_offset.get(
                    value & PAGE_MASK, ()
                ):
                    export_votes[value - offset] += 1
            for votes in (self_votes, export_votes):
                votes.pop(0, None)
            best = self._base_verdict(self_votes, export_votes) + (word,)
            if best[0] is not None:
                break  # the preferred word size has a verdict
        base, how, word = best
        if base is None:
            raise KsymtabNotFoundException(
                'No clear base address from the image pointers (%s); pass '
                % how
                + '--base-address'
            )
        if self.is_64_bits is None:
            self.is_64_bits = word == 8
        return base, how

    @staticmethod
    def _base_verdict(self_votes, export_votes):
        """(base or None, description of the votes)"""

        def ranked(votes):
            top = votes.most_common(2) + [(None, 0), (None, 0)]
            return top[0], top[1][1]

        (self_top, self_count), self_next = ranked(self_votes)
        (export_top, export_count), export_next = ranked(export_votes)
        how = 'self pointers: %s with %d votes, runner-up %d; ' % (
            '%#x' % self_top if self_top is not None else 'none',
            self_count,
            self_next,
        ) + 'pointers to exported symbols: %s with %d votes, runner-up %d' % (
            '%#x' % export_top if export_top is not None else 'none',
            export_count,
            export_next,
        )

        def clear(count, runner_up):
            return (
                count >= MIN_BASE_VOTES
                and count >= BASE_VOTE_RATIO * runner_up
            )

        if (
            self_top == export_top
            and self_count >= MIN_BASE_VOTES
            and export_count >= MIN_BASE_VOTES
        ):
            return self_top, how
        if clear(self_count, self_next) and export_top in (None, self_top):
            return self_top, how
        if clear(export_count, export_next) and self_top in (None, export_top):
            return export_top, how
        return None, how

    # -- symbols and sections ----------------------------------------------

    def make_symbols(self):
        base = self.kernel_text_candidate
        mask = (1 << (64 if self.is_64_bits else 32)) - 1
        self.symbols = []
        self.name_to_symbol = {}
        for name, value in self.entries():
            if name in self.name_to_symbol:
                continue
            symbol = KallsymsSymbol()
            symbol.name = name
            if self.ksymtab['kind'] == 'prel32':
                symbol.file_offset = value
                symbol.virtual_address = (base + value) & mask
            else:
                symbol.virtual_address = value
                symbol.file_offset = value - base
            symbol.symbol_type = KallsymsSymbolType.DATA
            symbol.is_global = True
            self.symbols.append(symbol)
            self.name_to_symbol[name] = symbol
        self.symbols.sort(key=lambda symbol: symbol.virtual_address)
        self.ksymtab['symbols'] = len(self.symbols)

    def find_layout(self):
        """Sections of the raw image (see the module docstring), and the
        types of the symbols from them."""
        base = self.kernel_text_candidate
        size = len(self.kernel_img)
        found = self.ksymtab
        tables_start = min(run.start for run in found['runs'])
        rodata, rodata_by = tables_start, 'start of __ksymtab'
        banner = BANNER_RE.search(self.kernel_img)
        if banner and banner.start() < tables_start:
            rodata, rodata_by = banner.start(), 'kernel banner'
        data = found['table'].end
        self.layout = [
            ('.text', base, base + rodata, 'AX'),
            ('.rodata', base + rodata, base + data, 'A'),
            ('.data', base + data, base + size, 'AW'),
        ]
        logging.info(
            '[+] Sections without kallsyms: .text %#x-%#x (up to the %s), '
            % (base, base + rodata, rodata_by)
            + '.rodata up to the end of __ksymtab_strings, then .data '
            + '%#x-%#x (init and data not told apart)'
            % (base + data, base + size)
        )
        text_end = base + rodata
        for symbol in self.symbols:
            if base <= symbol.virtual_address < text_end:
                symbol.symbol_type = KallsymsSymbolType.TEXT
            elif base + data <= symbol.virtual_address < base + size:
                symbol.symbol_type = KallsymsSymbolType.DATA
            elif base + rodata <= symbol.virtual_address < base + data:
                symbol.symbol_type = KallsymsSymbolType.RODATA
            elif symbol.virtual_address >= base + size:
                symbol.symbol_type = KallsymsSymbolType.BSS
            else:
                symbol.symbol_type = KallsymsSymbolType.ABSOLUTE

    def summary(self) -> dict:
        """What was found, as plain JSON-able values."""
        found = self.ksymtab
        return {
            'entries': found['count'],
            'layout': found['kind'],
            'entry_size': found['stride'],
            'big_endian': found['big_endian'],
            'bits': 64 if self.is_64_bits else 32,
            'tables': [
                ['%#x' % run.start, '%#x' % run.end] for run in found['runs']
            ],
            'strings': [
                '%#x' % found['table'].start,
                '%#x' % found['table'].end,
            ],
            'symbols': len(self.symbols),
            'functions': sum(
                symbol.symbol_type == KallsymsSymbolType.TEXT
                for symbol in self.symbols
            ),
            'base': '%#x' % self.kernel_text_candidate,
            'base_from': found.get('base_from'),
            'relocations': len(self.elf64_rela or ()),
            'sections': [
                [name, '%#x' % start, '%#x' % end, flags]
                for name, start, end, flags in (self.layout or ())
            ],
        }

    def print_symbols_debug(self, out_buffer=None):
        for symbol in self.symbols:
            out_string = '{0:s} {1:s} {2:s}'.format(
                '%016x' % symbol.virtual_address
                if self.is_64_bits
                else '%08x' % symbol.virtual_address,
                symbol.symbol_type.value,
                symbol.name,
            )
            if out_buffer:
                out_buffer.write(out_string + '\n')
            else:
                print(out_string)


def replace_kernel_sections(kernel, ksymtab_finder, file_contents):
    """
    Replace the ".kernel" PROGBITS section(s) of `kernel` by the
    sections KsymtabFinder could tell apart (its `layout`), keeping
    any hole punched for an arm64 relocation table.
    """
    old = [
        section
        for section in kernel.sections
        if section.section_name in ('.kernel', '.kernel2')
    ]
    if not old or not ksymtab_finder.layout:
        return
    base = old[0].section_header.sh_addr
    hole = None
    if ksymtab_finder.elf64_rela:
        hole = (
            base + ksymtab_finder.elf64_rela_start,
            base + ksymtab_finder.elf64_rela_end_excl,
        )
    flag_bits = {
        'A': SH_FLAGS.SHF_ALLOC,
        'W': SH_FLAGS.SHF_WRITE,
        'X': SH_FLAGS.SHF_EXECINSTR,
    }
    new = []
    for name, start, stop, flags in ksymtab_finder.layout:
        pieces = [(start, stop)]
        if hole and start < hole[1] and hole[0] < stop:
            pieces = [
                (a, b) for a, b in ((start, hole[0]), (hole[1], stop)) if b > a
            ]
        for number, (a, b) in enumerate(pieces):
            section = ElfProgbits(kernel)
            section.section_name = name + (
                '.%d' % (number + 1) if number else ''
            )
            section.section_header.sh_flags = 0
            for flag in flags:
                section.section_header.sh_flags |= flag_bits[flag]
            section.section_header.sh_addr = a
            section.section_header.sh_size = b - a
            section.section_contents = bytearray(
                file_contents[a - base : b - base]
            )
            new.append(section)
    index = kernel.sections.index(old[0])
    kernel.sections = [
        section for section in kernel.sections if section not in old
    ]
    kernel.sections[index:index] = new
    for section in kernel.sections:
        if isinstance(section, ElfNoBits):
            section.section_header.sh_flags = (
                SH_FLAGS.SHF_ALLOC | SH_FLAGS.SHF_WRITE
            )
