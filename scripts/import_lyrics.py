#!/usr/bin/env python3
"""Import existing lyrics documents into Side-Step ``.txt`` sidecars.

Built for messy sources: per-song files, album documents holding several
songs, and Suno-style prompts with ``[Verse]`` / ``[Chorus]`` tags.

Two stages, deliberately.  ``scan`` reads your documents, splits album
files into songs, matches them to audio files and writes a JSON mapping.
You review that mapping.  ``apply`` then writes the sidecars.  Nothing
touches a sidecar until you have looked at the matches.

Usage:
    # 1. build the mapping
    python scripts/import_lyrics.py scan \
        --audio-dir my_audio --lyrics-dir ~/lyrics -o lyrics_map.json

    # 2. read lyrics_map.json, fix any bad matches, set "skip": true to drop one

    # 3. preview, then write
    python scripts/import_lyrics.py apply lyrics_map.json
    python scripts/import_lyrics.py apply lyrics_map.json --write

Merging is non-destructive: existing ``bpm`` / ``key`` / ``caption`` from
audio analysis are preserved (policy ``overwrite_lyrics``), and
``write_sidecar`` keeps a ``.txt.bak`` of the previous version.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# scripts/ lives next to the package — make it importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
DOC_EXTS = {".txt", ".md", ".text"}

# A match below this is reported but never auto-applied.
MATCH_THRESHOLD = 0.72
# If the top two candidates are this close, the match is called ambiguous.
AMBIGUITY_MARGIN = 0.08


# ── Title detection ────────────────────────────────────────────────

_TITLE_PATTERNS = (
    re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$"),                       # markdown heading
    re.compile(r"^\*\*(?P<title>.+?)\*\*\s*$"),                        # **bold**
    re.compile(r"^\s*(?:track\s*)?\d{1,2}\s*[.)\-–—:]\s*(?P<title>\S.*?)\s*$", re.I),
    re.compile(r"^\s*title\s*[:\-]\s*(?P<title>.+?)\s*$", re.I),
)

# Structure tags belong to the lyrics — never treat them as titles.
_STRUCTURE_TAG = re.compile(r"^\s*[\[(](?:intro|verse|pre-?chorus|chorus|bridge|hook|outro|refrain|drop|break|instrumental|solo)", re.I)

# A bare ALL-CAPS line is a common album-doc title convention.
_ALLCAPS = re.compile(r"^[A-Z0-9][A-Z0-9 '’\-!?&,.()]{2,60}$")

# Suno style headers: comma-heavy descriptor lines, no sentence punctuation.
_STYLE_HEADER = re.compile(r"^[^.!?]*,[^.!?]*,[^.!?]*$")


def _looks_like_title(line: str) -> Optional[str]:
    """Return the extracted title if *line* reads as a song heading."""
    stripped = line.strip()
    if not stripped or _STRUCTURE_TAG.match(stripped):
        return None
    for pat in _TITLE_PATTERNS:
        m = pat.match(stripped)
        if m:
            title = m.group("title").strip(" *_#")
            return title or None
    if _ALLCAPS.match(stripped) and len(stripped.split()) <= 8:
        return stripped.title()
    return None


def _looks_like_style_header(line: str) -> bool:
    """Suno prompts often lead with 'industrial, dark synth, male vocals'."""
    stripped = line.strip()
    if not stripped or _STRUCTURE_TAG.match(stripped):
        return False
    return bool(_STYLE_HEADER.match(stripped)) and len(stripped) < 200


# ── Document splitting ─────────────────────────────────────────────

def split_document(path: Path) -> List[Dict[str, Any]]:
    """Split one document into ``[{title, lyrics, notes}]``.

    A document with no detectable headings yields a single song titled
    after the filename — the common per-song-file case.
    """
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()

    songs: List[Dict[str, Any]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    def _flush() -> None:
        if current_title is None and not any(l.strip() for l in current_lines):
            return
        body = "\n".join(current_lines).strip("\n")
        if not body.strip():
            return
        songs.append({
            "title": current_title or path.stem,
            "lyrics": body,
            "source": str(path),
        })

    for line in lines:
        title = _looks_like_title(line)
        if title is not None:
            _flush()
            current_title = title
            current_lines = []
            continue
        current_lines.append(line.rstrip())

    _flush()

    # Trim blank padding and flag Suno style headers for review.
    for song in songs:
        body_lines = song["lyrics"].split("\n")
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        notes: List[str] = []
        if body_lines and _looks_like_style_header(body_lines[0]):
            notes.append(f"possible Suno style header on line 1: {body_lines[0][:60]!r}")
        song["lyrics"] = "\n".join(body_lines)
        song["notes"] = notes

    return songs


# ── Matching ───────────────────────────────────────────────────────

_TRACK_PREFIX = re.compile(r"^\s*\d{1,2}\s*[.)\-_–—]\s*")


def _norm(name: str) -> str:
    """Normalise a title or filename stem for fuzzy comparison."""
    s = _TRACK_PREFIX.sub("", name.lower())
    s = s.replace("&", " and ").replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def match_songs(
    songs: List[Dict[str, Any]],
    audio_files: List[Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Match parsed songs to audio files. Returns ``(entries, leftovers)``."""
    norm_audio = {af: _norm(af.stem) for af in audio_files}
    used: Dict[Path, Dict[str, Any]] = {}
    leftovers: List[Dict[str, Any]] = []

    for song in songs:
        target = _norm(song["title"])
        scored = sorted(
            (
                (difflib.SequenceMatcher(None, target, na).ratio(), af)
                for af, na in norm_audio.items()
            ),
            key=lambda t: t[0],
            reverse=True,
        )
        if not scored:
            leftovers.append(song)
            continue

        best_score, best_audio = scored[0]
        notes = list(song.get("notes", []))
        status = "matched"

        if best_score < MATCH_THRESHOLD:
            leftovers.append({**song, "best_guess": best_audio.name, "score": round(best_score, 3)})
            continue
        if len(scored) > 1 and (best_score - scored[1][0]) < AMBIGUITY_MARGIN:
            status = "ambiguous"
            notes.append(f"close runner-up: {scored[1][1].name} ({scored[1][0]:.2f})")

        prior = used.get(best_audio)
        if prior is not None:
            # Two songs claimed the same audio file — keep the better one.
            if prior["score"] >= best_score:
                leftovers.append({**song, "best_guess": best_audio.name, "score": round(best_score, 3)})
                continue
            leftovers.append({k: prior[k] for k in ("title", "lyrics", "source")})
            notes.append(f"displaced weaker match {prior['title']!r}")

        entry = {
            "audio": str(best_audio),
            "sidecar": str(best_audio.with_suffix(".txt")),
            "title": song["title"],
            "source": song["source"],
            "score": round(best_score, 3),
            "status": status,
            "notes": notes,
            "skip": False,
            "lyrics": song["lyrics"],
        }
        used[best_audio] = entry

    entries = sorted(used.values(), key=lambda e: e["audio"])
    unmatched_audio = [af for af in audio_files if af not in used]
    return entries, leftovers + [
        {"title": None, "source": None, "unmatched_audio": str(af)} for af in unmatched_audio
    ]


# ── Commands ───────────────────────────────────────────────────────

def cmd_scan(args: argparse.Namespace) -> int:
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    lyrics_dir = Path(args.lyrics_dir).expanduser().resolve()

    if not audio_dir.is_dir():
        print(f"[FAIL] Not a directory: {audio_dir}")
        return 1
    if not lyrics_dir.is_dir():
        print(f"[FAIL] Not a directory: {lyrics_dir}")
        return 1

    audio_files = sorted(
        p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS
    )
    docs = sorted(p for p in lyrics_dir.rglob("*") if p.suffix.lower() in DOC_EXTS)

    if not audio_files:
        print(f"[FAIL] No audio files under {audio_dir}")
        return 1
    if not docs:
        print(f"[FAIL] No .txt/.md documents under {lyrics_dir}")
        return 1

    songs: List[Dict[str, Any]] = []
    for doc in docs:
        found = split_document(doc)
        songs.extend(found)
        print(f"  {doc.name}: {len(found)} song(s)")

    entries, leftovers = match_songs(songs, audio_files)

    out = Path(args.output).expanduser()
    out.write_text(
        json.dumps(
            {
                "audio_dir": str(audio_dir),
                "lyrics_dir": str(lyrics_dir),
                "entries": entries,
                "review": leftovers,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    matched = sum(1 for e in entries if e["status"] == "matched")
    ambiguous = sum(1 for e in entries if e["status"] == "ambiguous")
    flagged = sum(1 for e in entries if e["notes"])

    print()
    print(f"  audio files:     {len(audio_files)}")
    print(f"  songs parsed:    {len(songs)}")
    print(f"  matched:         {matched}")
    print(f"  ambiguous:       {ambiguous}  <- check these")
    print(f"  with notes:      {flagged}")
    print(f"  needs review:    {len(leftovers)}")
    print()
    print(f"[OK] Wrote {out}")
    print("     Review it, then: python scripts/import_lyrics.py apply "
          f"{out} --write")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    from sidestep_engine.data.sidecar_io import (
        merge_fields, read_sidecar, write_sidecar,
    )

    map_path = Path(args.mapping).expanduser()
    if not map_path.is_file():
        print(f"[FAIL] No such mapping file: {map_path}")
        return 1

    data = json.loads(map_path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if not entries:
        print("[FAIL] Mapping contains no entries.")
        return 1

    dry_run = not args.write
    written = skipped = 0

    for entry in entries:
        sidecar = Path(entry["sidecar"])
        lyrics = entry.get("lyrics", "").strip()

        if entry.get("skip"):
            skipped += 1
            continue
        if not lyrics:
            print(f"  SKIP (empty lyrics)   {sidecar.name}")
            skipped += 1
            continue
        if entry.get("status") == "ambiguous" and not args.include_ambiguous:
            print(f"  SKIP (ambiguous)      {sidecar.name}  <- --include-ambiguous to write")
            skipped += 1
            continue

        existing = read_sidecar(sidecar)
        merged = merge_fields(existing, {"lyrics": lyrics}, policy="overwrite_lyrics")
        if not args.no_instrumental:
            # This track has lyrics, so it is definitively not instrumental.
            # overwrite_all only touches the keys present in new_fields, so
            # this corrects a wrong value without disturbing caption/bpm/key.
            merged = merge_fields(merged, {"is_instrumental": "false"}, policy="overwrite_all")

        had = "replacing" if existing.get("lyrics", "").strip() else "adding"
        n_lines = len(lyrics.splitlines())
        print(f"  {'WOULD WRITE' if dry_run else 'WRITE'}  {sidecar.name}  "
              f"({had} lyrics, {n_lines} lines, keeping "
              f"{len([k for k in existing if k != 'lyrics'])} existing field(s))")

        if not dry_run:
            write_sidecar(sidecar, merged)
        written += 1

    # Audio with no matching lyrics document is instrumental by the user's
    # rule. Worth stating explicitly: a caption pass may have written a
    # wrong is_instrumental, and an explicit value beats dataset_builder's
    # inference from empty lyrics.
    instrumental = 0
    if not args.no_instrumental:
        for item in data.get("review", []):
            audio = item.get("unmatched_audio")
            if not audio:
                continue
            sidecar = Path(audio).with_suffix(".txt")
            existing = read_sidecar(sidecar)
            if existing.get("lyrics", "").strip():
                continue  # has lyrics from somewhere else — leave it alone
            merged = merge_fields(
                existing, {"is_instrumental": "true"}, policy="overwrite_all",
            )
            was = existing.get("is_instrumental", "").strip() or "unset"
            print(f"  {'WOULD MARK' if dry_run else 'MARK'}  {sidecar.name}  "
                  f"instrumental (was: {was})")
            if not dry_run:
                write_sidecar(sidecar, merged)
            instrumental += 1

    print()
    if instrumental:
        print(f"{'Would mark' if dry_run else 'Marked'} {instrumental} track(s) instrumental.")
    if dry_run:
        print(f"[DRY RUN] {written} sidecar(s) would be written, {skipped} skipped.")
        print("          Re-run with --write to apply.")
    else:
        print(f"[OK] {written} sidecar(s) written, {skipped} skipped.")
        print("     Previous versions saved as .txt.bak")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import lyrics documents into Side-Step sidecars.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="parse lyrics docs and build a mapping")
    p_scan.add_argument("--audio-dir", required=True, help="directory of audio files")
    p_scan.add_argument("--lyrics-dir", required=True, help="directory of lyrics documents")
    p_scan.add_argument("-o", "--output", default="lyrics_map.json", help="mapping file to write")
    p_scan.set_defaults(func=cmd_scan)

    p_apply = sub.add_parser("apply", help="write sidecars from a reviewed mapping")
    p_apply.add_argument("mapping", help="mapping JSON produced by scan")
    p_apply.add_argument("--write", action="store_true", help="actually write (default: dry run)")
    p_apply.add_argument("--include-ambiguous", action="store_true",
                         help="also write entries flagged ambiguous")
    p_apply.add_argument("--no-instrumental", action="store_true",
                         help="do not set is_instrumental (default: audio with no "
                              "lyrics file is marked instrumental, audio with lyrics false)")
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
