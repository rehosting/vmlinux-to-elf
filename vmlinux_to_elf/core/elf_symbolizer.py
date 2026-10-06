#!/usr/bin/env python3
# -*- encoding: Utf-8 -*-
import logging
from io import BytesIO

from vmlinux_to_elf.core.architecture_detecter import ArchitectureGuessError
from vmlinux_to_elf.core.kallsyms import KallsymsFinder, KallsymsSymbolType
from vmlinux_to_elf.core.auto_unpack import Signature
from vmlinux_to_elf.core import sections as sections_mod
from vmlinux_to_elf.utils.elf import (
    SH_FLAGS,
    SPECIAL_SECTION_INDEX,
    ST_INFO_BINDING,
    ST_INFO_TYPE,
    Elf32BigEndianRelocationWithAddendTableEntry,
    Elf32BigEndianSymbolTableEntry,
    Elf32LittleEndianRelocationWithAddendTableEntry,
    Elf32LittleEndianSymbolTableEntry,
    Elf64BigEndianRelocationWithAddendTableEntry,
    Elf64BigEndianSymbolTableEntry,
    Elf64LittleEndianRelocationWithAddendTableEntry,
    Elf64LittleEndianSymbolTableEntry,
    ElfFile,
    ElfNoBits,
    ElfNullSection,
    ElfProgbits,
    ElfRela,
    ElfStrtab,
    ElfSymtab,
)

"""
    The ElfSymbolizer class, defined in this file, gathers information from
    the other modules (such as kallsyms_finder, which extracts the kernel's
    runtime symbol table, or vmlinuz_decompressor, which processes possible
    kernel compressions), in order to generate the output ELF file.
"""


class ElfSymbolizer:
    def __init__(
        self,
        file_contents: bytes,
        output_file: str = None,
        output_stream: BytesIO = None,
        elf_machine: int = None,
        bit_size: int = None,
        base_address: int = None,
        bss_size: int = 16,
        file_offset: int = None,
        override_relative: bool = None,
        # extra_info: bool = False,
        split_sections: bool = False,
    ):
        """
            split_sections: cut the raw kernel into head/text/rodata/init/
            data sections at its linker symbols (core.sections) instead of
            one RWX ".kernel", and give every symbol the size of the gap to
            the next symbol in its section, typed FUNC in executable sections
            and OBJECT elsewhere. For an ELF input, its own sections are kept
            and only the symbols are sized and typed that way.
        """

        if file_contents.startswith(
            Signature.uImage.value
        ):  # uImage header magic (always big-endian)
            if not file_offset:
                file_offset = 64  # uImage header size (image_header_t from u-boot/image.h)

            if not base_address:
                base_address = int.from_bytes(
                    file_contents[4 * 4 : 4 * 5], 'big'
                )

        if file_offset:
            file_contents = file_contents[file_offset:]

        kallsyms_finder = KallsymsFinder(
            file_contents,
            bit_size,
            override_relative,
            base_address,
            # extra_info,
        )

        if elf_machine is None and not kallsyms_finder.elf_machine:
            raise ArchitectureGuessError(
                'The architecture could not be guessed successfully'
            )

        # If we got an ELF file input, retain
        # the interesting metadata

        if file_contents.startswith(b'\x7fELF'):
            kernel = ElfFile.from_bytes(BytesIO(file_contents))

        else:
            kernel = ElfFile(
                kallsyms_finder.is_big_endian, kallsyms_finder.is_64_bits
            )

            # Previously the register size was based on the kernel
            # version string:
            #  bool(kallsyms_finder.offset_table_element_size >= 8 or
            #    search('itanium|(?:amd|aarch|ia|arm|x86_|\D-)64',
            #    kallsyms_finder.version_string, flags = IGNORECASE))

            if elf_machine is not None:
                kernel.file_header.e_machine = elf_machine
            else:
                kernel.file_header.e_machine = kallsyms_finder.elf_machine

            ET_EXEC = 2
            kernel.file_header.e_type = ET_EXEC

            null = ElfNullSection(kernel)
            null.section_name = ''

            progbits = ElfProgbits(kernel)
            progbits.section_name = '.kernel'
            progbits.section_header.sh_flags = (
                SH_FLAGS.SHF_ALLOC
                | SH_FLAGS.SHF_EXECINSTR
                | SH_FLAGS.SHF_WRITE
            )

            if base_address is not None:
                progbits.section_header.sh_addr = base_address
                logging.info(
                    f'[+] Reconstructing ELF with provided base address ({progbits.section_header.sh_addr:x})'
                )
            else:
                progbits.section_header.sh_addr = (
                    kallsyms_finder.kernel_text_candidate
                )
                logging.info(
                    f'[+] Reconstructing ELF with guessed base address ({progbits.section_header.sh_addr:x})'
                )

            kernel.sections += [null, progbits]

            if kallsyms_finder.elf64_rela:
                # Punch a hole into the ELF to remove relocation tables
                progbits.section_header.sh_size = (
                    kallsyms_finder.elf64_rela_start
                )
                progbits.section_contents = bytearray(
                    file_contents[: progbits.section_header.sh_size]
                )
                progbits2 = ElfProgbits(kernel)
                progbits2.section_name = '.kernel2'
                progbits2.section_header.sh_flags = (
                    SH_FLAGS.SHF_ALLOC
                    | SH_FLAGS.SHF_EXECINSTR
                    | SH_FLAGS.SHF_WRITE
                )
                progbits2.section_header.sh_addr = (
                    progbits.section_header.sh_addr
                    + kallsyms_finder.elf64_rela_end_excl
                )
                progbits2.section_header.sh_size = (
                    len(file_contents) - kallsyms_finder.elf64_rela_end_excl
                )
                progbits2.section_contents = bytearray(
                    file_contents[kallsyms_finder.elf64_rela_end_excl :]
                )
                kernel.sections += [progbits2]
            else:
                progbits.section_contents = bytearray(file_contents)
                progbits.section_header.sh_size = len(file_contents)

            bss = ElfNoBits(kernel)
            bss.section_name = '.bss'
            bss.section_header.sh_flags = (
                SH_FLAGS.SHF_ALLOC
                | SH_FLAGS.SHF_EXECINSTR
                | SH_FLAGS.SHF_WRITE
            )
            bss.section_header.sh_size = bss_size * 1024 * 1024
            bss.section_header.sh_addr = progbits.section_header.sh_addr + len(
                file_contents
            )

            kernel.sections += [bss]

            if split_sections:
                self.split_kernel_sections(
                    kernel, kallsyms_finder, file_contents, bss
                )

        r"""
            Find the entry point symbol. Based on executing this command
            on the Linux tree source:
            
            for i in $(find -iname 'vmlinux.lds.S' -o -iname 'dyn.lds.S' -o -iname 'vmlinux-std.lds');
                do echo "$i:"$(grep -P '^ENTRY\(' $i);
            done | grep -Po 'ENTRY\((.+?)\)' | sort -u

            You can find the possible symbols that are used as an entry
            point for the kernel, here sorted from the most specific to
            the less specific
        """

        POSSIBLE_ENTRY_POINT_SYMBOLS = [
            'kernel_entry',
            'microblaze_start',
            'parisc_kernel_start',
            'phys_startup_32',
            'phys_startup_64',
            'phys_start',
            '_stext_lma',
            'res_service',
            '_c_int00',
            'startup_32',
            'startup_64',
            'startup_continue',
            'startup',
            '__start',
            '_start',
            'start_kernel',
            'stext',
            '_stext',
            '_text',
        ]

        entry_point_address: int = None

        for symbol_name in POSSIBLE_ENTRY_POINT_SYMBOLS:
            symbol = kallsyms_finder.name_to_symbol.get(symbol_name)

            if symbol:
                entry_point_address = symbol.virtual_address

                break

        if entry_point_address is None:
            raise ValueError('No entry point symbol found in the kallsyms')

        kernel.file_header.e_entry = entry_point_address

        # Add symbols

        symtab = next(
            (i for i in kernel.sections if i.section_name == '.symtab'), None
        )

        if not symtab:
            symtab = ElfSymtab(kernel)
            symtab.section_name = '.symtab'

            strtab = ElfStrtab(kernel)
            strtab.section_name = '.strtab'
            symtab.string_table = strtab

            shstrtab = ElfStrtab(kernel)
            shstrtab.section_name = '.shstrtab'

            kernel.symbol_table = symtab
            kernel.section_string_table = shstrtab
            kernel.sections += [symtab, strtab, shstrtab]

        elf_symbol_class = {
            (False, False): Elf32LittleEndianSymbolTableEntry,
            (True, False): Elf32BigEndianSymbolTableEntry,
            (False, True): Elf64LittleEndianSymbolTableEntry,
            (True, True): Elf64BigEndianSymbolTableEntry,
        }[(kernel.is_big_endian, kernel.is_64_bits)]

        symbol_sizes = {}
        if split_sections:
            symbol_sizes = sections_mod.symbol_sizes(
                [symbol.virtual_address for symbol in kallsyms_finder.symbols],
                [
                    (
                        section.section_header.sh_addr,
                        section.section_header.sh_addr
                        + section.section_header.sh_size,
                    )
                    for section in kernel.sections
                    if section.section_header.sh_flags & SH_FLAGS.SHF_ALLOC
                    and not isinstance(section, ElfNoBits)
                ],
            )

        for symbol in kallsyms_finder.symbols:
            elf_symbol = elf_symbol_class(
                kernel.is_big_endian, kernel.is_64_bits
            )

            elf_symbol.symbol_name = symbol.name
            elf_symbol.st_value = symbol.virtual_address

            if symbol.symbol_type not in (
                KallsymsSymbolType.TEXT,
                KallsymsSymbolType.WEAK_SYMBOL_WITH_DEFAULT,
            ):
                elf_symbol.st_info_type = ST_INFO_TYPE.STT_OBJECT
            else:
                elf_symbol.st_info_type = ST_INFO_TYPE.STT_FUNC

            if symbol.symbol_type in (
                KallsymsSymbolType.WEAK_OBJECT_WITH_DEFAULT,
                KallsymsSymbolType.WEAK_SYMBOL_WITH_DEFAULT,
            ):
                elf_symbol.st_info_binding = ST_INFO_BINDING.STB_WEAK
            elif symbol.is_global:
                elf_symbol.st_info_binding = ST_INFO_BINDING.STB_GLOBAL
            else:
                elf_symbol.st_info_binding = ST_INFO_BINDING.STB_LOCAL

            if symbol.symbol_type == KallsymsSymbolType.ABSOLUTE:
                elf_symbol.st_shndx = SPECIAL_SECTION_INDEX.SHN_ABS
            else:
                elf_symbol.associated_section = kernel.find_section(
                    symbol.virtual_address
                )
                if split_sections:
                    elf_symbol.st_size = symbol_sizes.get(
                        symbol.virtual_address, 0
                    )
                    section = elf_symbol.associated_section
                    if section is not None and not isinstance(
                        section, ElfNoBits
                    ):
                        elf_symbol.st_info_type = (
                            ST_INFO_TYPE.STT_FUNC
                            if section.section_header.sh_flags
                            & SH_FLAGS.SHF_EXECINSTR
                            else ST_INFO_TYPE.STT_OBJECT
                        )

            symtab.symbol_table.append(elf_symbol)

        if kallsyms_finder.elf64_rela:
            srela = ElfRela(kernel)
            srela.section_name = '.rela.dyn'
            relocation_class = {
                (
                    False,
                    False,
                ): Elf32LittleEndianRelocationWithAddendTableEntry,
                (True, False): Elf32BigEndianRelocationWithAddendTableEntry,
                (False, True): Elf64LittleEndianRelocationWithAddendTableEntry,
                (True, True): Elf64BigEndianRelocationWithAddendTableEntry,
            }[(kernel.is_big_endian, kernel.is_64_bits)]
            srela.relocation_table = []
            srela.symtab_section = symtab
            kernel.sections += [srela]
            for rela in kallsyms_finder.elf64_rela:
                relocation = relocation_class(
                    kernel.is_big_endian, kernel.is_64_bits
                )

                R_AARCH64_RELATIVE = 0x403

                relocation.r_offset = rela[0]
                relocation.r_info_type = R_AARCH64_RELATIVE
                relocation.r_addend = rela[2]

                srela.relocation_table.append(relocation)

        # Save the modified ELF

        if output_file:
            with open(output_file, 'wb') as fd:
                kernel.serialize(fd)

            logging.info(
                '[+] Successfully wrote the new ELF kernel to %s' % output_file
            )

        else:
            kernel.serialize(output_stream)

    @staticmethod
    def split_kernel_sections(kernel, kallsyms_finder, file_contents, bss):
        """
            Replace the ".kernel" PROGBITS section(s) of `kernel` by the
            layout from core.sections, keeping any hole punched for an
            arm64 relocation table.
        """
        old = [
            section
            for section in kernel.sections
            if section.section_name in ('.kernel', '.kernel2')
        ]
        base = old[0].section_header.sh_addr
        end = base + len(file_contents)
        symbols = {
            name: symbol.virtual_address
            for name, symbol in kallsyms_finder.name_to_symbol.items()
        }
        try:
            layout, bounds = sections_mod.kernel_sections(
                symbols, base, end, kernel.file_header.e_machine
            )
        except sections_mod.SectionLayoutError as error:
            logging.warning(
                f'[!] Keeping a single .kernel section: {error}'
            )
            return

        hole = None
        if kallsyms_finder.elf64_rela:
            hole = (
                base + kallsyms_finder.elf64_rela_start,
                base + kallsyms_finder.elf64_rela_end_excl,
            )

        flag_bits = {
            'A': SH_FLAGS.SHF_ALLOC,
            'W': SH_FLAGS.SHF_WRITE,
            'X': SH_FLAGS.SHF_EXECINSTR,
        }
        new = []
        for (name, start, stop, flags), (start_by, stop_by) in zip(
            layout, bounds
        ):
            pieces = [(start, stop)]
            if hole and start < hole[1] and hole[0] < stop:
                pieces = [
                    (a, b)
                    for a, b in ((start, hole[0]), (hole[1], stop))
                    if b > a
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
            logging.info(
                f'[+] Section {name}: {start:x}-{stop:x} '
                + f'({start_by} .. {stop_by})'
            )

        index = kernel.sections.index(old[0])
        kernel.sections = [
            section for section in kernel.sections if section not in old
        ]
        kernel.sections[index:index] = new
        bss.section_header.sh_flags = SH_FLAGS.SHF_ALLOC | SH_FLAGS.SHF_WRITE
