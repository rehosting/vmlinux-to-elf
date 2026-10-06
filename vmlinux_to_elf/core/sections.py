#!/usr/bin/env python3
# -*- encoding: Utf-8 -*-
"""
    Section layout of a raw kernel image, from the linker symbols that
    kallsyms keeps even without CONFIG_KALLSYMS_ALL (_text, _stext,
    __start_rodata, _etext, __init_begin, _sinittext, _einittext).

    Used by ElfSymbolizer(split_sections=True): instead of one RWX ".kernel"
    section, the output ELF gets head/text/rodata/init/data sections with the
    right flags, so that disassemblers and loaders can tell code from data,
    and every symbol can be given the size of the gap to the next symbol in
    its section.

    Layout families:

    * text first, init after (arm64, ARM >= 3.x, MIPS, x86): .head.text up to
      _stext, .text up to __start_rodata (or _etext), .rodata up to _etext,
      then init text. MIPS places data between _etext and __init_begin (so
      that span is .data, and what follows init is .init.data); the others
      place rodata there and data after init.
    * init first (ARM 2.6: _sinittext at the image base, stext inside init):
      .init.text, .init.data up to _text, then text/rodata/data.

    Without __start_rodata, .text/.rodata is cut at the lowest symbol whose
    name only read-only data carries (linux_banner, __param_str_*, ...)
    below _etext; otherwise .text runs to _etext.
"""
from typing import Dict, List, Optional, Tuple

EM_MIPS = 8

RODATA_NAMES = {'linux_banner', 'linux_proc_banner'}
RODATA_PREFIXES = ('__param_str_', '__func__.', '__kstrtab_', '__ksymtab_')

# (name, start, end, flags) with flags a subset of 'AWX'
Section = Tuple[str, int, int, str]


class SectionLayoutError(Exception):
    pass


def _first(symbols: Dict[str, int], *names) -> Tuple[Optional[str], Optional[int]]:
    for name in names:
        if name in symbols:
            return name, symbols[name]
    return None, None


def kernel_sections(
    symbols: Dict[str, int], base: int, end: int, elf_machine: int
) -> Tuple[List[Section], List[Tuple[str, str]]]:
    """
        Sections covering [base, end) as (name, start, end, flags), plus
        for each the symbols (or addresses) that bound it.
    """
    inside = {n: v for n, v in symbols.items() if base <= v <= end}
    text_name, text = _first(inside, '_stext', 'stext', '_text')
    etext = inside.get('_etext')
    init_name, init = _first(inside, '__init_begin', '_sinittext')
    einittext = inside.get('_einittext')

    missing = [
        name
        for name, value in (
            ('_stext/_text', text),
            ('_etext', etext),
            ('__init_begin/_sinittext', init),
            ('_einittext', einittext),
        )
        if value is None
    ]
    if missing:
        raise SectionLayoutError(
            'kallsyms lacks the linker symbols ' + ', '.join(missing)
        )

    init_first = (
        init <= text < einittext and inside.get('_text', 0) >= einittext
    )
    if init_first:  # ARM 2.6: stext is the head code, inside init
        text_name, text = '_text', inside['_text']

    sections: List[Section] = []
    bounds: List[Tuple[str, str]] = []

    def add(name, start, stop, flags, start_by, stop_by):
        if stop <= start:
            return
        if sections and sections[-1][0] == name and sections[-1][2] == start:
            # e.g. rodata running up to a page-aligned __init_begin
            prev = sections.pop()
            prev_by = bounds.pop()
            start, start_by = prev[1], prev_by[0]
        sections.append((name, start, stop, flags))
        bounds.append((start_by, stop_by))

    def text_and_rodata(start, start_by):
        rodata = inside.get('__start_rodata')
        rodata_by = '__start_rodata'
        if rodata is None or not start < rodata < etext:
            rodata, rodata_by = None, None
            markers = [
                (v, n)
                for n, v in inside.items()
                if start < v < etext
                and (n in RODATA_NAMES or n.startswith(RODATA_PREFIXES))
            ]
            if markers:
                rodata, name = min(markers)
                rodata_by = name + ' (lowest read-only-data name)'
        if rodata is None:
            add('.text', start, etext, 'AX', start_by, '_etext')
        else:
            add('.text', start, rodata, 'AX', start_by, rodata_by)
            add('.rodata', rodata, etext, 'A', rodata_by, '_etext')

    if init_first:
        add('.head.text', base, init, 'AX', '%#x' % base, init_name)
        add('.init.text', init, einittext, 'AX', init_name, '_einittext')
        add('.init.data', einittext, text, 'AW', '_einittext', text_name)
        text_and_rodata(text, text_name)
        add('.data', etext, end, 'AW', '_etext', 'end')
        return sections, bounds

    add('.head.text', base, text, 'AX', '%#x' % base, text_name)
    text_and_rodata(text, text_name)
    if init <= etext:  # ARM 3.x: init text right after _etext
        add('.init.text', etext, einittext, 'AX', '_etext', '_einittext')
        add('.data', einittext, end, 'AW', '_einittext', 'end')
    elif elf_machine == EM_MIPS:  # rodata + data, then init
        add('.data', etext, init, 'AW', '_etext', init_name)
        add('.init.text', init, einittext, 'AX', init_name, '_einittext')
        add('.init.data', einittext, end, 'AW', '_einittext', 'end')
    else:  # rodata, init, data
        add('.rodata', etext, init, 'A', '_etext', init_name)
        add('.init.text', init, einittext, 'AX', init_name, '_einittext')
        add('.data', einittext, end, 'AW', '_einittext', 'end')
    return sections, bounds


def symbol_sizes(
    addresses: List[int], sections: List[Tuple[int, int]]
) -> Dict[int, int]:
    """
        Size of the symbol at each address: the distance to the next
        higher symbol address in the same section, or to the section end.
        Addresses outside every section get no entry.
    """
    sizes = {}
    ordered = sorted(set(addresses))
    for start, stop in sections:
        inside = [a for a in ordered if start <= a < stop]
        for i, address in enumerate(inside):
            nxt = inside[i + 1] if i + 1 < len(inside) else stop
            sizes[address] = nxt - address
    return sizes
