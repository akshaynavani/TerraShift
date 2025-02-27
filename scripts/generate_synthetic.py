"""Generate a synthetic bi-temporal dataset for development."""
from __future__ import annotations

import argparse
from collections import Counter

from terrashift.config import DATA_DIR, SEED, TILE_SIZE
from terrashift.data.synthetic import generate_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DATA_DIR / "synthetic"))
    p.add_argument("--n-per-class", type=int, default=200)
    p.add_argument("--size", type=int, default=TILE_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    records = generate_dataset(args.out, args.n_per_class, args.size, args.seed)
    print(f"Wrote {len(records)} tile pairs to {args.out}")
    print(dict(Counter(r.label for r in records)))


if __name__ == "__main__":
    main()
