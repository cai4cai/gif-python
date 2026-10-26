"""Download and unpack the Mindboggle-101 volumes from Harvard Dataverse.

Data set: Klein et al., "101 labeled brain images and a consistent human cortical labeling protocol",
https://doi.org/10.7910/DVN/HMQKCK. The five volume archives are public and need no login. They are
verified against ``checksums.sha256`` (shipped in ``src/resources/mindboggle101``) before anything is
extracted; the data are not hosted by this project.

Target layout (``<atlas_dir>`` is ``data/atlases/Mindboggle101`` by default)::

    <atlas_dir>/<subject>/t1weighted.nii.gz
    <atlas_dir>/<subject>/t1weighted_brain.nii.gz
    <atlas_dir>/<subject>/labels.DKT31.manual+aseg.nii.gz
    <atlas_dir>/<subject>/labels.DKT31.manual+aseg_cleaned.nii.gz     (with clean=True)
    <atlas_dir>/structures_info.csv, tissues_info.csv, itksnap_label_descriptions_mindboggle101.txt
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Dict, Iterable, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

DOI = "10.7910/DVN/HMQKCK"
DATAFILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"

#: files that are extracted by default (the ones used by the pipeline)
DEFAULT_FILES = ("t1weighted.nii.gz", "t1weighted_brain.nii.gz", "labels.DKT31.manual+aseg.nii.gz")
LABELS_NAME = "labels.DKT31.manual+aseg.nii.gz"
CLEANED_LABELS_NAME = "labels.DKT31.manual+aseg_cleaned.nii.gz"
RESOURCE_FILES = ("structures_info.csv", "tissues_info.csv", "itksnap_label_descriptions_mindboggle101.txt")
#: sub-folder prefix of the cleaned-label entries in checksums.sha256
CLEANED_PREFIX = "cleaned-labels/"
#: number of subjects of the full data set
EXPECTED_SUBJECTS = 101

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


@dataclass(frozen=True)
class Archive:
    """One volume archive of the Dataverse data set."""

    name: str  #: short name, e.g. ``OASIS-TRT-20``
    file_id: int  #: Dataverse data file id
    size: int  #: size in bytes (from the Dataverse metadata)
    n_subjects: int

    @property
    def filename(self) -> str:
        return f"{self.name}_volumes.tar.gz"

    @property
    def url(self) -> str:
        return DATAFILE_URL.format(file_id=self.file_id)


ARCHIVES: Dict[str, Archive] = {
    a.name: a
    for a in (
        Archive("OASIS-TRT-20", 3392553, 408405660, 20),
        Archive("NKI-TRT-20", 3392550, 435004408, 20),
        Archive("NKI-RS-22", 3392551, 924552880, 22),
        Archive("MMRR-21", 3392552, 843015849, 21),
        Archive("Extra-18", 3392549, 732012448, 18),
    )
}


class ChecksumError(RuntimeError):
    """A downloaded or generated file does not match its recorded SHA-256."""


# --------------------------------------------------------------------------------- checksums
def sha256_file(path: PathLike) -> str:
    """SHA-256 of the bytes of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_label_volume(path: PathLike) -> str:
    """SHA-256 of the voxel values of a NIfTI label volume.

    The hash covers the data type, the shape and the voxel values in C order, not the header and not the
    compression, so it does not depend on the nibabel or zlib version that wrote the file.
    """
    import nibabel as nib
    import numpy as np

    data = np.asanyarray(nib.load(str(path)).dataobj)
    digest = hashlib.sha256()
    digest.update(f"{data.dtype.name}{data.shape}".encode())
    digest.update(np.ascontiguousarray(data).astype(data.dtype.newbyteorder("<"), copy=False).tobytes())
    return digest.hexdigest()


def parse_checksums(text: str) -> Dict[str, str]:
    """Parse ``<sha256>  <name>`` lines (``sha256sum`` format); ``#`` comments and blank lines are ignored."""
    checksums: Dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if not match:
            raise ValueError(f"checksums line {number} is not '<sha256>  <name>': {line!r}")
        checksums[match.group(2)] = match.group(1).lower()
    return checksums


def format_checksums(checksums: Dict[str, str]) -> str:
    return "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items()))


def load_shipped_checksums() -> Dict[str, str]:
    """The checksums shipped with the package (archives and cleaned labels)."""
    from src.resources import mindboggle101_file

    return parse_checksums(mindboggle101_file("checksums.sha256").read_text())


# --------------------------------------------------------------------------------- download
def default_opener(url: str) -> BinaryIO:
    request = urllib.request.Request(url, headers={"User-Agent": "gif-python fetch-mindboggle"})
    return urllib.request.urlopen(request, timeout=60)


def download_archive(
    archive: Archive,
    download_dir: PathLike,
    expected_sha256: Optional[str],
    opener: Optional[Callable[[str], BinaryIO]] = None,
    progress: bool = True,
) -> Path:
    """Download ``archive`` into ``download_dir`` unless a verified copy is already there.

    :param expected_sha256: SHA-256 to verify against (``None`` skips verification)
    :param opener: function that opens a URL for reading (default: :func:`default_opener`)
    :raises ChecksumError: if the downloaded file does not match; the file is removed
    """
    opener = opener or default_opener
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    target = download_dir / archive.filename
    if target.is_file():
        if expected_sha256 is None or sha256_file(target) == expected_sha256:
            logger.info("%s already downloaded", archive.filename)
            return target
        logger.warning("%s exists but does not match its checksum; downloading it again", target)
        target.unlink()

    part = target.with_suffix(target.suffix + ".part")
    logger.info("Downloading %s (%.0f MB) from %s", archive.filename, archive.size / 1e6, archive.url)
    digest = hashlib.sha256()
    try:
        with opener(archive.url) as response, part.open("wb") as out:
            bar = None
            if progress:
                from tqdm import tqdm

                bar = tqdm(total=archive.size, unit="B", unit_scale=True, desc=archive.name)
            for chunk in iter(lambda: response.read(1 << 20), b""):
                out.write(chunk)
                digest.update(chunk)
                if bar:
                    bar.update(len(chunk))
            if bar:
                bar.close()
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise ChecksumError(
                f"SHA-256 of the downloaded {archive.filename} is {digest.hexdigest()}, expected {expected_sha256}. "
                "The download is corrupt or the file at Harvard Dataverse changed."
            )
        part.replace(target)
    finally:
        if part.exists():
            part.unlink()
    return target


# --------------------------------------------------------------------------------- extraction
def extract_volumes(
    archive_path: PathLike,
    atlas_dir: PathLike,
    files: Optional[Sequence[str]] = DEFAULT_FILES,
) -> List[str]:
    """Extract ``<top>/<subject>/<file>`` members of a volume archive into ``atlas_dir/<subject>/<file>``.

    Only regular files with plain names are written, nothing else of the archive is used (no links, no
    absolute or ``..`` paths). ``files=None`` extracts every regular file whose name does not start with a dot.

    :return: the subject names that were extracted
    """
    atlas_dir = Path(atlas_dir)
    subjects: List[str] = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if not member.isreg():
                continue
            parts = member.name.split("/")
            if len(parts) != 3:
                continue
            _, subject, filename = parts
            if not (_SAFE_NAME.match(subject) and _SAFE_NAME.match(filename)):
                continue
            if files is not None and filename not in files:
                continue
            destination = atlas_dir / subject / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, destination.open("wb") as out:
                shutil.copyfileobj(source, out)
            if subject not in subjects:
                subjects.append(subject)
    return subjects


def copy_label_tables(atlas_dir: PathLike) -> List[Path]:
    """Copy the shipped csv and ITK-SNAP files into the atlas folder."""
    from src.resources import mindboggle101_file

    atlas_dir = Path(atlas_dir)
    atlas_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in RESOURCE_FILES:
        destination = atlas_dir / name
        shutil.copyfile(mindboggle101_file(name), destination)
        copied.append(destination)
    return copied


# --------------------------------------------------------------------------------- cleaning
def clean_atlas_labels(
    atlas_dir: PathLike,
    subjects: Iterable[str],
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, str]:
    """Run ``clean-labels`` on every subject and return ``{"cleaned-labels/<subject>/<file>": checksum}``."""
    from src.labels.cleaning import DEFAULT_SPARSE_LABELS, clean_labels_file

    atlas_dir = Path(atlas_dir)
    labels = list(labels) if labels else list(DEFAULT_SPARSE_LABELS)
    checksums: Dict[str, str] = {}
    for subject in sorted(subjects):
        source = atlas_dir / subject / LABELS_NAME
        if not source.is_file():
            logger.warning("%s has no %s; not cleaned", subject, LABELS_NAME)
            continue
        out, n_changed = clean_labels_file(source, atlas_dir / subject / CLEANED_LABELS_NAME, labels)
        logger.info("%s: %d voxels changed", subject, n_changed)
        checksums[f"{CLEANED_PREFIX}{subject}/{CLEANED_LABELS_NAME}"] = sha256_label_volume(out)
    return checksums


def verify_cleaned(checksums: Dict[str, str], shipped: Dict[str, str]) -> List[str]:
    """Compare the cleaned-label checksums with the shipped ones.

    :return: names whose checksum differs from the shipped one (names without a shipped entry are only logged)
    """
    mismatches = []
    missing = []
    for name, digest in checksums.items():
        if name not in shipped:
            missing.append(name)
        elif shipped[name] != digest:
            mismatches.append(name)
    if missing:
        logger.info("No shipped checksum for %d cleaned label file(s)", len(missing))
    return mismatches


# --------------------------------------------------------------------------------- driver
@dataclass
class FetchResult:
    """What :func:`fetch_mindboggle` did."""

    subjects: List[str]
    archive_checksums: Dict[str, str]  #: SHA-256 of the archives that were fetched, by file name
    cleaned_checksums: Dict[str, str]  #: checksums of the cleaned labels (empty without ``clean``)


def fetch_mindboggle(
    atlas_dir: PathLike,
    download_dir: PathLike,
    archives: Optional[Sequence[str]] = None,
    files: Optional[Sequence[str]] = DEFAULT_FILES,
    clean: bool = False,
    clean_labels: Optional[Sequence[int]] = None,
    verify: bool = True,
    delete_archives: bool = False,
    opener: Optional[Callable[[str], BinaryIO]] = None,
    progress: bool = True,
) -> FetchResult:
    """Download, verify and unpack the Mindboggle-101 volumes; optionally clean the labels.

    :param archives: names of the archives to fetch (default: all five, i.e. all 101 subjects)
    :param files: files to extract per subject (``None``: all)
    :param clean: also write ``labels.DKT31.manual+aseg_cleaned.nii.gz`` and verify/write their checksums
    :param verify: check the SHA-256 of the archives (and cleaned labels) against ``checksums.sha256``
    :return: the extracted subjects and the checksums that were computed
    :raises ChecksumError: on a mismatch
    :raises ValueError: for an unknown archive name, or an archive whose checksum is not recorded yet
    """
    names = list(archives) if archives else list(ARCHIVES)
    unknown = [n for n in names if n not in ARCHIVES]
    if unknown:
        raise ValueError(f"Unknown archive(s) {unknown}; choose from {list(ARCHIVES)}")
    shipped = load_shipped_checksums()
    atlas_dir = Path(atlas_dir)

    subjects: List[str] = []
    archive_checksums: Dict[str, str] = {}
    cleaned: Dict[str, str] = {}
    for name in names:
        archive = ARCHIVES[name]
        expected = shipped.get(archive.filename)
        if verify and expected is None:
            raise ValueError(
                f"No SHA-256 for {archive.filename} is recorded in checksums.sha256; "
                "re-run with --skip-verify to use the download unverified"
            )
        path = download_archive(archive, download_dir, expected if verify else None, opener, progress)
        archive_checksums[archive.filename] = expected if verify else sha256_file(path)
        extracted = extract_volumes(path, atlas_dir, files)
        logger.info("%s: extracted %d subjects", name, len(extracted))
        subjects.extend(extracted)
        if delete_archives:
            path.unlink()
    copy_label_tables(atlas_dir)

    if clean:
        cleaned = clean_atlas_labels(atlas_dir, subjects, clean_labels)
        record = atlas_dir / "cleaned_labels.sha256"
        record.write_text(
            "# SHA-256 of the voxel values (data type, shape, C-order little-endian data) of the cleaned labels,\n"
            "# as produced by `gif-python clean-labels`. Compare with src/resources/mindboggle101/checksums.sha256.\n"
            + format_checksums(cleaned)
        )
        if verify and clean_labels is None:
            mismatches = verify_cleaned(cleaned, shipped)
            if mismatches:
                raise ChecksumError(
                    f"{len(mismatches)} cleaned label file(s) differ from the shipped checksums, e.g. {mismatches[0]}"
                )
    return FetchResult(subjects, archive_checksums, cleaned)
