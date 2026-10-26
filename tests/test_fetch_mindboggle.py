import hashlib
import io
import tarfile

import nibabel as nib
import numpy as np
import pytest

import src.datasets.mindboggle as mb
from src.cli.main import main
from src.datasets.mindboggle import (
    ARCHIVES,
    Archive,
    ChecksumError,
    download_archive,
    extract_volumes,
    fetch_mindboggle,
    format_checksums,
    parse_checksums,
    sha256_file,
    sha256_label_volume,
)


def label_volume(sparse_label=None):
    data = np.full((9, 9, 9), 17, dtype=np.float32)
    data[:, :, :4] = 3
    if sparse_label:
        data[4, 4, 5] = sparse_label
    return data


def add_bytes(tar, name, payload: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def nifti_bytes(data, tmp_path, name="x.nii.gz"):
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path.read_bytes()


@pytest.fixture
def fake_archive(tmp_path):
    """A small archive laid out like the Dataverse ones, with junk and malicious members."""
    path = tmp_path / "serve" / "Fake-2_volumes.tar.gz"
    path.parent.mkdir()
    with tarfile.open(path, "w:gz") as tar:
        add_bytes(tar, "Fake-2_volumes/.DS_Store", b"junk")
        for index, sparse in ((1, 1033), (2, None)):
            subject = f"Fake-2-{index}"
            add_bytes(
                tar,
                f"Fake-2_volumes/{subject}/t1weighted.nii.gz",
                nifti_bytes(np.ones((9, 9, 9), np.float32), tmp_path),
            )
            add_bytes(
                tar,
                f"Fake-2_volumes/{subject}/labels.DKT31.manual+aseg.nii.gz",
                nifti_bytes(label_volume(sparse), tmp_path),
            )
            add_bytes(tar, f"Fake-2_volumes/{subject}/labels.DKT31.manual.nii.gz", b"not needed")
            add_bytes(tar, f"Fake-2_volumes/{subject}/._t1weighted.nii.gz", b"resource fork")
        add_bytes(tar, "Fake-2_volumes/../../escape.txt", b"bad")
        add_bytes(tar, "/abs/path.txt", b"bad")
        add_bytes(tar, "Fake-2_volumes/Fake-2-1/sub/deeper.nii.gz", b"bad")
        link = tarfile.TarInfo("Fake-2_volumes/Fake-2-1/labels.link.nii.gz")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
    return path


@pytest.fixture
def patched(monkeypatch, fake_archive):
    archive = Archive("Fake-2", 99, fake_archive.stat().st_size, 2)
    monkeypatch.setattr(mb, "ARCHIVES", {"Fake-2": archive})
    checksums = {archive.filename: sha256_file(fake_archive)}
    monkeypatch.setattr(mb, "load_shipped_checksums", lambda: dict(checksums))
    monkeypatch.setattr(mb, "default_opener", lambda url: io.BytesIO(fake_archive.read_bytes()))
    return archive, checksums


def opener_for(path, calls):
    def opener(url):
        calls.append(url)
        return open(path, "rb")

    return opener


def test_checksum_file_format_roundtrip_and_errors():
    digest = "ab" * 32
    text = f"# comment\n\n{digest}  a.tar.gz\n{digest} *b/c.nii\n"
    assert parse_checksums(text) == {"a.tar.gz": digest, "b/c.nii": digest}
    assert parse_checksums(format_checksums({"x": digest})) == {"x": digest}
    with pytest.raises(ValueError, match="line 1"):
        parse_checksums("not a checksum line\n")


def test_shipped_checksums_cover_the_five_archives_and_all_cleaned_labels():
    shipped = mb.load_shipped_checksums()
    for archive in ARCHIVES.values():
        assert len(shipped[archive.filename]) == 64
    cleaned = {name: digest for name, digest in shipped.items() if name.startswith(mb.CLEANED_PREFIX)}
    assert len(cleaned) == mb.EXPECTED_SUBJECTS == sum(a.n_subjects for a in ARCHIVES.values()) + 0
    assert all(name.endswith(mb.CLEANED_LABELS_NAME) for name in cleaned)
    assert len(set(shipped.values())) > len(ARCHIVES)  # the cleaned volumes differ between subjects


def test_archive_urls_follow_the_dataverse_pattern():
    assert ARCHIVES["OASIS-TRT-20"].url == "https://dataverse.harvard.edu/api/access/datafile/3392553"
    assert {a.file_id for a in ARCHIVES.values()} == {3392553, 3392550, 3392551, 3392552, 3392549}


def test_download_verifies_and_skips_valid_copies(tmp_path, fake_archive):
    archive = Archive("Fake-2", 99, fake_archive.stat().st_size, 2)
    digest = sha256_file(fake_archive)
    calls = []
    target = download_archive(archive, tmp_path / "dl", digest, opener_for(fake_archive, calls), progress=False)
    assert target.read_bytes() == fake_archive.read_bytes() and len(calls) == 1
    download_archive(archive, tmp_path / "dl", digest, opener_for(fake_archive, calls), progress=False)
    assert len(calls) == 1  # verified copy is reused
    target.write_bytes(b"corrupt")
    download_archive(archive, tmp_path / "dl", digest, opener_for(fake_archive, calls), progress=False)
    assert len(calls) == 2 and sha256_file(target) == digest


def test_download_with_wrong_checksum_leaves_nothing_behind(tmp_path, fake_archive):
    archive = Archive("Fake-2", 99, fake_archive.stat().st_size, 2)
    with pytest.raises(ChecksumError, match="expected"):
        download_archive(archive, tmp_path / "dl", "0" * 64, opener_for(fake_archive, []), progress=False)
    assert list((tmp_path / "dl").iterdir()) == []


def test_extraction_is_restricted_to_the_wanted_regular_files(tmp_path, fake_archive):
    atlas = tmp_path / "atlas"
    subjects = extract_volumes(fake_archive, atlas)
    assert sorted(subjects) == ["Fake-2-1", "Fake-2-2"]
    written = sorted(p.relative_to(atlas).as_posix() for p in atlas.rglob("*") if p.is_file())
    assert written == [
        "Fake-2-1/labels.DKT31.manual+aseg.nii.gz",
        "Fake-2-1/t1weighted.nii.gz",
        "Fake-2-2/labels.DKT31.manual+aseg.nii.gz",
        "Fake-2-2/t1weighted.nii.gz",
    ]
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path.parent / "escape.txt").exists()


def test_extract_all_files_still_ignores_dot_files_and_unsafe_names(tmp_path, fake_archive):
    atlas = tmp_path / "atlas"
    extract_volumes(fake_archive, atlas, files=None)
    assert (atlas / "Fake-2-1" / "labels.DKT31.manual.nii.gz").is_file()
    assert not (atlas / "Fake-2-1" / "._t1weighted.nii.gz").exists()
    assert not (atlas / "Fake-2-1" / "labels.link.nii.gz").exists()


def test_label_checksum_ignores_header_and_compression(tmp_path):
    data = label_volume(1033)
    a = tmp_path / "a.nii.gz"
    b = tmp_path / "b.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(a))
    image = nib.Nifti1Image(data, np.diag([2, 2, 2, 1.0]))
    image.header["descrip"] = b"different header"
    nib.save(image, str(b))
    assert sha256_label_volume(a) == sha256_label_volume(b)
    other = tmp_path / "c.nii.gz"
    nib.save(nib.Nifti1Image(label_volume(None), np.eye(4)), str(other))
    assert sha256_label_volume(other) != sha256_label_volume(a)
    changed_dtype = tmp_path / "d.nii.gz"
    nib.save(nib.Nifti1Image(data.astype(np.int16), np.eye(4)), str(changed_dtype))
    assert sha256_label_volume(changed_dtype) != sha256_label_volume(a)


def test_fetch_downloads_verifies_extracts_and_copies_tables(tmp_path, patched):
    result = fetch_mindboggle(tmp_path / "atlases", tmp_path / "dl", progress=False)
    assert sorted(result.subjects) == ["Fake-2-1", "Fake-2-2"]
    assert result.archive_checksums == patched[1]
    for name in mb.RESOURCE_FILES:
        assert (tmp_path / "atlases" / name).is_file()
    assert (tmp_path / "dl" / "Fake-2_volumes.tar.gz").is_file()
    assert result.cleaned_checksums == {}


def test_fetch_clean_writes_cleaned_labels_and_checks_the_shipped_checksums(tmp_path, patched, monkeypatch):
    archive, checksums = patched
    # first run without shipped cleaned checksums: records them
    result = fetch_mindboggle(tmp_path / "atlases", tmp_path / "dl", clean=True, progress=False)
    cleaned = nib.load(str(tmp_path / "atlases" / "Fake-2-1" / mb.CLEANED_LABELS_NAME)).get_fdata()
    assert 1033 not in cleaned and cleaned[4, 4, 5] == 17
    key = f"{mb.CLEANED_PREFIX}Fake-2-1/{mb.CLEANED_LABELS_NAME}"
    assert key in result.cleaned_checksums
    record = parse_checksums((tmp_path / "atlases" / "cleaned_labels.sha256").read_text())
    assert record == result.cleaned_checksums
    # with matching shipped checksums the run passes, with a wrong one it fails
    monkeypatch.setattr(mb, "load_shipped_checksums", lambda: {**checksums, **result.cleaned_checksums})
    fetch_mindboggle(tmp_path / "atlases2", tmp_path / "dl", clean=True, progress=False)
    monkeypatch.setattr(mb, "load_shipped_checksums", lambda: {**checksums, key: "0" * 64})
    with pytest.raises(ChecksumError, match="cleaned"):
        fetch_mindboggle(tmp_path / "atlases3", tmp_path / "dl", clean=True, progress=False)


def test_fetch_checksum_mismatch_of_the_download_stops_before_extraction(tmp_path, patched, monkeypatch):
    archive, _ = patched
    monkeypatch.setattr(mb, "load_shipped_checksums", lambda: {archive.filename: "f" * 64})
    with pytest.raises(ChecksumError):
        fetch_mindboggle(tmp_path / "atlases", tmp_path / "dl", progress=False)
    assert not (tmp_path / "atlases").exists()


def test_fetch_requires_a_recorded_checksum_unless_verification_is_skipped(tmp_path, patched, monkeypatch):
    monkeypatch.setattr(mb, "load_shipped_checksums", lambda: {})
    with pytest.raises(ValueError, match="skip-verify"):
        fetch_mindboggle(tmp_path / "atlases", tmp_path / "dl", progress=False)
    result = fetch_mindboggle(tmp_path / "atlases", tmp_path / "dl", verify=False, progress=False)
    assert len(result.subjects) == 2 and "Fake-2_volumes.tar.gz" in result.archive_checksums


def test_fetch_unknown_archive_and_delete_option(tmp_path, patched):
    with pytest.raises(ValueError, match="Unknown archive"):
        fetch_mindboggle(tmp_path / "a", tmp_path / "dl", archives=["Nope"], progress=False)
    fetch_mindboggle(tmp_path / "a", tmp_path / "dl", delete_archives=True, progress=False)
    assert not (tmp_path / "dl" / "Fake-2_volumes.tar.gz").exists()


def test_cli_list(capsys):
    assert main(["fetch-mindboggle", "--list"]) == 0
    out = capsys.readouterr().out
    assert "OASIS-TRT-20" in out and "3392553" in out and "(not recorded)" not in out


def test_cli_fetch_with_fake_download(tmp_path, patched, capsys):
    code = main(
        [
            "fetch-mindboggle",
            "--data-dir",
            str(tmp_path / "data"),
            "--clean",
            "--write-checksums",
            str(tmp_path / "sums.sha256"),
        ]
    )
    assert code == 0
    assert "2 subjects" in capsys.readouterr().out
    assert (tmp_path / "data" / "atlases" / "Mindboggle101" / "Fake-2-1" / mb.CLEANED_LABELS_NAME).is_file()
    written = parse_checksums((tmp_path / "sums.sha256").read_text())
    assert "Fake-2_volumes.tar.gz" in written and any(k.startswith("cleaned-labels/") for k in written)


def test_sha256_file(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()
