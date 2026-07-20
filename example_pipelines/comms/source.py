"""Bind the shared SigMF reader to communications dataset discovery."""

from pathlib import Path

from sigvue.plugin import DirectorySource

from ..io.sigmf import describe_recording, load_recording


def recording_source(root: Path) -> DirectorySource:
    return DirectorySource(
        root,
        pattern="*.sigmf-meta",
        loader=load_recording,
        describe=describe_recording,
        recursive=True,
    )
