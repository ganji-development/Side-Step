#!/usr/bin/env python3
"""Import lyrics files into Side-Step ``.txt`` sidecars.

One file per song.  The filename is the song title, every line in the
file is lyrics.  Nothing is parsed, split, stripped or interpreted — the
text goes in exactly as written.

Two stages, deliberately.  ``scan`` matches lyrics files to audio files
by name and writes a JSON mapping.  You review it.  ``apply`` then
writes the sidecars.  Nothing is touched until you have seen the matches.

Audio with no matching lyrics file is marked ``is_instrumental: true``.

Usage:
    # 1. build the mapping
    python scripts/import_lyrics.py scan \
        --audio-dir my_audio --lyrics-dir lyrics -o lyrics_map.json

    # 2. read lyrics_map.json, fix bad matches, set "skip": true to drop one

    # 3. preview, then write
    python scripts/import_lyrics.py apply lyrics_map.json
    python scripts/import_lyrics.py apply lyrics_map.json --write

Merging is non-destructive: existing ``bpm`` / ``key`` / ``caption`` from
audio analysis and captioning are preserved (policy ``overwrite_lyrics``),
and ``write_sidecar`` keeps a ``.txt.bak`` of the previous version.
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# scripts/ lives next to the package — make it importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
DOC_EXTS = {".txt", ".md", ".text"}

# A match below this is reported but never auto-applied.
MATCH_THRESHOLD = 0.72
# If the top two candidates are this close, the match is called ambiguous.
AMBIGUITY_MARGIN = 0.08


def read_lyrics_file(path: Path) -> Dict[str, Any]:
    """Read one lyrics file whole. Title comes from the filename."""
    body = path.read_text(encoding="utf-8-sig", errors="replace")
    return {
        "title": path.stem,
        "lyrics": body.strip("\n"),
        "source": str(path),
    }


# ── Matching ───────────────────────────────────────────────────────

_TRACK_PREFIX = re.compile(r"^\s*\d{1,2}\s*[.)\-_–—]\s*")


def _norm(name: str) -> str:
    """Normalise a filename stem for fuzzy comparison.

    Handles case, punctuation and unicode dashes so that
    ``Vine-Code Breach.txt`` still finds ``Vine‑Code Breach.wav``.
    """
    s = _TRACK_PREFIX.sub("", name.lower())
    s = s.replace("&", " and ").replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def match_songs(
    songs: List[Dict[str, Any]],
    audio_files: List[Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Match lyrics files to audio files. Returns ``(entries, leftovers)``."""
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
        notes: List[str] = []
        status = "matched"

        if best_score < MATCH_THRESHOLD:
            leftovers.append({**song, "best_guess": best_audio.name, "score": round(best_score, 3)})
            continue
        if len(scored) > 1 and (best_score - scored[1][0]) < AMBIGUITY_MARGIN:
            status = "ambiguous"
            notes.append(f"close runner-up: {scored[1][1].name} ({scored[1][0]:.2f})")

        prior = used.get(best_audio)
        if prior is not None:
            # Two lyrics files claimed the same audio — keep the better match.
            if prior["score"] >= best_score:
                leftovers.append({**song, "best_guess": best_audio.name, "score": round(best_score, 3)})
                continue
            leftovers.append({k: prior[k] for k in ("title", "lyrics", "source")})
            notes.append(f"displaced weaker match {prior['title']!r}")

        used[best_audio] = {
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

    audio_files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    docs = sorted(p for p in lyrics_dir.rglob("*") if p.suffix.lower() in DOC_EXTS)

    if not audio_files:
        print(f"[FAIL] No audio files under {audio_dir}")
        return 1
    if not docs:
        print(f"[FAIL] No .txt/.md files under {lyrics_dir}")
        return 1

    songs = [read_lyrics_file(doc) for doc in docs]
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
    unmatched_lyrics = sum(1 for r in leftovers if r.get("title"))
    instrumental = sum(1 for r in leftovers if r.get("unmatched_audio"))

    print(f"  lyrics files:      {len(docs)}")
    print(f"  audio files:       {len(audio_files)}")
    print(f"  matched:           {matched}")
    if ambiguous:
        print(f"  ambiguous:         {ambiguous}  <- check these")
    if unmatched_lyrics:
        print(f"  lyrics unmatched:  {unmatched_lyrics}  <- name mismatch, check these")
    print(f"  -> instrumental:   {instrumental}  (audio with no lyrics file)")
    print()
    print(f"[OK] Wrote {out}")
    print(f"     Review it, then: python scripts/import_lyrics.py apply {out} --write")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    from sidestep_engine.data.sidecar_io import merge_fields, read_sidecar, write_sidecar

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
            # Has lyrics, so definitively not instrumental. overwrite_all only
            # visits keys present in new_fields, so this corrects a wrong value
            # without disturbing caption/bpm/key.
            merged = merge_fields(merged, {"is_instrumental": "false"}, policy="overwrite_all")

        had = "replacing" if existing.get("lyrics", "").strip() else "adding"
        kept = len([k for k in existing if k != "lyrics"])
        print(f"  {'WOULD WRITE' if dry_run else 'WRITE'}  {sidecar.name}  "
              f"({had} lyrics, {len(lyrics.splitlines())} lines, keeping {kept} field(s))")

        if not dry_run:
            write_sidecar(sidecar, merged)
        written += 1

    # Audio with no lyrics file is instrumental. Stated explicitly because a
    # caption pass may have written a wrong is_instrumental, and an explicit
    # value beats dataset_builder's inference from empty lyrics.
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
            merged = merge_fields(existing, {"is_instrumental": "true"}, policy="overwrite_all")
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
        description="Import lyrics files into Side-Step sidecars (one file per song).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="match lyrics files to audio and build a mapping")
    p_scan.add_argument("--audio-dir", required=True, help="directory of audio files")
    p_scan.add_argument("--lyrics-dir", required=True, help="directory of lyrics files")
    p_scan.add_argument("-o", "--output", default="lyrics_map.json", help="mapping file to write")
    p_scan.set_defaults(func=cmd_scan)

    p_apply = sub.add_parser("apply", help="write sidecars from a reviewed mapping")
    p_apply.add_argument("mapping", help="mapping JSON produced by scan")
    p_apply.add_argument("--write", action="store_true", help="actually write (default: dry run)")
    p_apply.add_argument("--include-ambiguous", action="store_true",
                         help="also write entries flagged ambiguous")
    p_apply.add_argument("--no-instrumental", action="store_true",
                         help="do not touch is_instrumental at all")
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
