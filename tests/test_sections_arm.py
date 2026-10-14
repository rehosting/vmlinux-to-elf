from vmlinux_to_elf.core.sections import kernel_sections, EM_ARM, EM_MIPS, SectionLayoutError
import pytest

SYMS = {'_stext': 0xc00081a0, '__start_rodata': 0xc041e000, 'linux_banner': 0xc041e0a0,
        '_etext': 0xc0540000, '_einittext': 0xc055f310}


def test_arm_init_starts_at_etext_without_init_begin():
    secs, _ = kernel_sections(SYMS, 0xc0008000, 0xc0600000, EM_ARM)
    names = [s[0] for s in secs]
    assert names == ['.head.text', '.text', '.rodata', '.init.text', '.data']
    init = secs[names.index('.init.text')]
    assert init[1:3] == (0xc0540000, 0xc055f310)


def test_other_arches_still_need_an_init_start():
    with pytest.raises(SectionLayoutError):
        kernel_sections(SYMS, 0xc0008000, 0xc0600000, EM_MIPS)
