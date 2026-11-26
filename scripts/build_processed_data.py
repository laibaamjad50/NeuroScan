"""Rebuild processed_data/ from the raw dataset and the Phase 1 manifest.

processed_data/ is gitignored (about 300 MB), so it is not in the repository.
Everyone needs it before they can train, and re-running the whole Phase 1
notebook just to get it is slow and regenerates figures nobody asked for.

This script reproduces it exactly. It applies the same transform as Section 7
of the Phase 1 notebook — greyscale, centre-pad to a square with black, resize
to 224x224 with LANCZOS, save as PNG — and writes each image to the very path
recorded in data/cleaned_manifest.csv, so the output matches what the notebook
produced, file for file.

Usage, from the repository root:

    python scripts/build_processed_data.py            # build what is missing
    python scripts/build_processed_data.py --force    # rebuild everything
    python scripts/build_processed_data.py --check    # verify only, write nothing

Needs only pandas and Pillow.
"""

from __future__ import annotations

import argparse
import os
import sys

IMG_SIZE = (224, 224)


def pad_to_square(img, fill: int = 0):
    """Centre the image on a square black canvas (as Phase 1 does)."""
    from PIL import Image

    w, h = img.size
    s = max(w, h)
    canvas = Image.new("L", (s, s), fill)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def preprocess(path: str):
    from PIL import Image

    img = Image.open(path).convert("L")
    return pad_to_square(img).resize(IMG_SIZE, Image.LANCZOS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="repository root (default: inferred from this file)")
    ap.add_argument("--force", action="store_true", help="rebuild images that already exist")
    ap.add_argument("--check", action="store_true", help="report status and exit without writing")
    args = ap.parse_args(argv)

    import pandas as pd

    manifest = os.path.join(args.root, "data", "cleaned_manifest.csv")
    if not os.path.exists(manifest):
        print(f"ERROR: {manifest} not found. Run the Phase 1 notebook first.", file=sys.stderr)
        return 2

    df = pd.read_csv(manifest)
    for col in ("processed_path", "original_path"):
        df[col] = df[col].str.replace("\\", "/", regex=False)

    present = sum(os.path.exists(os.path.join(args.root, p)) for p in df["processed_path"])
    print(f"manifest rows     : {len(df):,}")
    print(f"already on disk   : {present:,}")
    print(f"to build          : {len(df) - present if not args.force else len(df):,}")

    if args.check:
        missing_src = [p for p in df["original_path"]
                       if not os.path.exists(os.path.join(args.root, p))][:5]
        if missing_src:
            print(f"WARNING: source images missing, e.g. {missing_src}", file=sys.stderr)
        ok = present == len(df)
        print("status            :", "complete" if ok else "INCOMPLETE")
        return 0 if ok else 1

    try:
        from tqdm.auto import tqdm
    except ImportError:                                   # tqdm is optional
        def tqdm(it, **kw):
            return it

    built = skipped = 0
    failures: list[str] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Preprocessing"):
        dest = os.path.join(args.root, row.processed_path)
        if os.path.exists(dest) and not args.force:
            skipped += 1
            continue
        src = os.path.join(args.root, row.original_path)
        if not os.path.exists(src):
            failures.append(row.original_path)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            preprocess(src).save(dest)
            built += 1
        except Exception as exc:                          # noqa: BLE001 - report, keep going
            failures.append(f"{row.original_path}: {type(exc).__name__}: {exc}")

    print(f"\nbuilt {built:,} | skipped {skipped:,} | failed {len(failures):,}")
    if failures:
        print("failures (first 10):", file=sys.stderr)
        for f in failures[:10]:
            print("   ", f, file=sys.stderr)
        return 1

    final = sum(os.path.exists(os.path.join(args.root, p)) for p in df["processed_path"])
    print(f"processed_data/ now holds {final:,} of {len(df):,} expected images")
    return 0 if final == len(df) else 1


if __name__ == "__main__":
    raise SystemExit(main())
