#!/usr/bin/env python3
# -*- encoding: Utf-8 -*-
"""
Tests for core.ksymtab on synthetic kernel images: export tables in each
layout (absolute 32/64-bit, PREL32, with and without a namespace field)
and byte order, and the base address found without being given.

python3 -m unittest discover -s tests
"""

import struct
import unittest
from io import BytesIO
from unittest import mock

from vmlinux_to_elf.core import elf_symbolizer
from vmlinux_to_elf.core.kallsyms import KallsymsNotFoundException
from vmlinux_to_elf.core.ksymtab import (
    KsymtabFinder,
    KsymtabNotFoundException,
    flatten_elf,
)
from vmlinux_to_elf.utils.elf import ElfFile

EM_ARM, EM_AARCH64 = 40, 183

FUNCTIONS = ['export_function_%02d' % index for index in range(80)]
OBJECTS = ['export_object_%02d' % index for index in range(8)]


def build_image(
    base, word, big_endian, prel32, namespace, string_pointers=False
):
    """
    (image, {name: address}): text (64-byte functions), a banner,
    __ksymtab, __ksymtab_strings, then data holding the exported
    objects, self pointers and pointers to exported functions.
    """
    order = '>' if big_endian else '<'
    pointer = order + ('Q' if word == 8 else 'I')
    image = bytearray()
    addresses = {}
    for name in FUNCTIONS:
        addresses[name] = len(image)
        image += b'\x1f\x20\x03\xd5' * 16
    image += b'\0' * 64
    image += b'Linux version 5.4.0 (builder@host) (gcc version 9.3.0) #1\n\0'
    image += b'\0' * (-len(image) % 64)

    names = sorted(FUNCTIONS + OBJECTS)
    fields = 3 if namespace else 2
    entry_size = (4 if prel32 else word) * fields
    ksymtab = len(image)
    image += b'\0' * (entry_size * len(names))
    string_offsets = {}
    for name in names:
        string_offsets[name] = len(image)
        image += name.encode() + b'\0'
    namespace_offset = len(image)
    image += b'TEST_NS\0'
    image += b'\0' * (-len(image) % 4096)

    for name in OBJECTS:
        addresses[name] = len(image)
        image += b'\0' * 32
    # list heads pointing at themselves, and an ops table of functions
    for _ in range(32):
        image += struct.pack(pointer, base + len(image))
    for name in FUNCTIONS:
        image += struct.pack(pointer, base + addresses[name])
    if string_pointers:  # e.g. a table of names: not export entries
        for name in names:
            image += struct.pack(pointer, base + string_offsets[name])
    image += b'\0' * 256

    for index, name in enumerate(names):
        entry = ksymtab + index * entry_size
        ns = namespace_offset if index % 5 == 0 else None
        if prel32:
            values = [
                addresses[name] - entry,
                string_offsets[name] - (entry + 4),
            ]
            if namespace:
                values.append(0 if ns is None else ns - (entry + 8))
            struct.pack_into(order + 'i' * fields, image, entry, *values)
        else:
            values = [base + addresses[name], base + string_offsets[name]]
            if namespace:
                values.append(0 if ns is None else base + ns)
            struct.pack_into(
                order + ('Q' if word == 8 else 'I') * fields,
                image,
                entry,
                *values,
            )
    return bytes(image), {
        name: base + offset for name, offset in addresses.items()
    }


class KsymtabFinderTest(unittest.TestCase):
    def check(self, base, word, big_endian, prel32, namespace):
        image, addresses = build_image(
            base, word, big_endian, prel32, namespace
        )
        finder = KsymtabFinder(image, bit_size=word * 8)
        self.assertEqual(finder.kernel_text_candidate, base)
        self.assertEqual(finder.is_big_endian, big_endian)
        self.assertEqual(
            finder.ksymtab['kind'], 'prel32' if prel32 else 'absolute'
        )
        self.assertEqual(
            {symbol.name: symbol.virtual_address for symbol in finder.symbols},
            addresses,
        )
        functions = {
            symbol.name
            for symbol in finder.symbols
            if symbol.symbol_type.value == 'T'
        }
        self.assertEqual(functions, set(FUNCTIONS))
        names = [name for name, _, _, _ in finder.layout]
        self.assertEqual(names, ['.text', '.rodata', '.data'])
        self.assertEqual(finder.summary()['symbols'], len(addresses))
        return finder

    def test_prel32_namespace_little_endian(self):
        self.check(0xFFFF800010080000, 8, False, True, True)

    def test_prel32_big_endian(self):
        self.check(0xFFFF800010080000, 8, True, True, False)

    def test_absolute_32_big_endian(self):
        self.check(0x80100000, 4, True, False, False)

    def test_absolute_32_namespace_little_endian(self):
        self.check(0xC0008000, 4, False, False, True)

    def test_absolute_64_namespace(self):
        self.check(0xFFFFFFFF81000000, 8, False, False, True)

    def test_explicit_base(self):
        image, addresses = build_image(0x80008000, 4, False, True, False)
        finder = KsymtabFinder(
            image, bit_size=32, explicit_base_address=0x80008000
        )
        self.assertEqual(finder.ksymtab['base_from'], 'provided')
        self.assertEqual(
            finder.name_to_symbol['export_object_00'].virtual_address,
            addresses['export_object_00'],
        )

    def test_no_export_table(self):
        image = b'Linux version 5.4.0 (x) #1\n\0' + bytes(range(256)) * 64
        with self.assertRaises(KsymtabNotFoundException):
            KsymtabFinder(image, bit_size=32)

    def test_unrelocated_entries_are_refused(self):
        # a relocatable kernel whose absolute entries are only filled in at
        # boot: no guess from whatever else points into the strings
        image, _ = build_image(0x80008000, 4, True, False, True)
        image = bytearray(image)
        strings = image.index(b'export_function_00\0')
        ksymtab = strings - 12 * (len(FUNCTIONS) + len(OBJECTS))
        image[ksymtab:strings] = bytes(strings - ksymtab)
        with self.assertRaises(KsymtabNotFoundException):
            KsymtabFinder(bytes(image), bit_size=32)

    def test_array_of_string_pointers_is_not_a_ksymtab(self):
        image, addresses = build_image(
            0x80008000, 4, False, False, False, string_pointers=True
        )
        finder = KsymtabFinder(image, bit_size=32)
        self.assertEqual(len(finder.ksymtab['runs']), 1)
        self.assertEqual(
            {symbol.name: symbol.virtual_address for symbol in finder.symbols},
            addresses,
        )


class ElfSymbolizerTest(unittest.TestCase):
    def test_ksymtab_only(self):
        base = 0xFFFF800010080000
        image, addresses = build_image(base, 8, False, True, True)
        output = BytesIO()
        symbolizer = elf_symbolizer.ElfSymbolizer(
            image,
            output_stream=output,
            elf_machine=EM_AARCH64,
            bit_size=64,
            ksymtab_only=True,
        )
        self.assertEqual(symbolizer.symbol_source, 'ksymtab')
        self.assertEqual(symbolizer.ksymtab['base'], '%#x' % base)
        elf = ElfFile.from_bytes(BytesIO(output.getvalue()))
        sections = {
            section.section_name: section.section_header
            for section in elf.sections
        }
        self.assertIn('.text', sections)
        self.assertNotIn('.kernel', sections)
        symbols = {
            symbol.symbol_name: symbol
            for symbol in elf.symbol_table.symbol_table
            if symbol.symbol_name
        }
        self.assertEqual(
            {name: symbol.st_value for name, symbol in symbols.items()},
            addresses,
        )
        self.assertEqual(elf.file_header.e_entry, base)

    def test_fallback_only_when_asked(self):
        image, _ = build_image(0x80008000, 4, False, False, False)

        class NoKallsyms:
            def __init__(self, *args, **kwargs):
                raise KallsymsNotFoundException('no kallsyms here')

        with mock.patch.object(elf_symbolizer, 'KallsymsFinder', NoKallsyms):
            with self.assertRaises(KallsymsNotFoundException):
                elf_symbolizer.ElfSymbolizer(
                    image,
                    output_stream=BytesIO(),
                    elf_machine=EM_ARM,
                    bit_size=32,
                )
            symbolizer = elf_symbolizer.ElfSymbolizer(
                image,
                output_stream=BytesIO(),
                elf_machine=EM_ARM,
                bit_size=32,
                ksymtab_fallback=True,
            )
        self.assertEqual(symbolizer.symbol_source, 'ksymtab')

    def test_flatten_elf(self):
        image, addresses = build_image(0x80008000, 4, False, False, False)
        output = BytesIO()
        elf_symbolizer.ElfSymbolizer(
            image,
            output_stream=output,
            elf_machine=EM_ARM,
            bit_size=32,
            ksymtab_only=True,
        )
        flat, base = flatten_elf(output.getvalue())
        self.assertEqual(base, 0x80008000)
        self.assertEqual(flat[: len(image)], image)
        finder = KsymtabFinder(output.getvalue())
        self.assertEqual(
            {symbol.name: symbol.virtual_address for symbol in finder.symbols},
            addresses,
        )


if __name__ == '__main__':
    unittest.main()
