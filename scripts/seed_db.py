"""Seed the database with a synthetic scene and (optionally) run predictions on it."""
from __future__ import annotations

import argparse
from datetime import date

from sqlalchemy import select

from api import crud, models as m, schemas as s
from api.database import Base, SessionLocal, engine
from api.inference import get_inference
from api.settings import get_settings
from terrashift.data.synthetic import generate_dataset


def main() -> None:
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="synthetic-review-demo")
    p.add_argument("--subdir", default="review_demo", help="Directory under the imagery root to write tiles into")
    p.add_argument("--n-per-class", type=int, default=20)
    p.add_argument("--predict", action="store_true", help="Run the trained models on every tile")
    args = p.parse_args()

    Base.metadata.create_all(bind=engine)
    root = settings.terrashift_imagery_root
    if not (root / args.subdir / "manifest.json").exists():
        generate_dataset(root / args.subdir, n_per_class=args.n_per_class, seed=7)

    with SessionLocal() as db:
        scene = db.scalar(select(m.Scene).where(m.Scene.name == args.scene))
        if scene is None:
            scene = crud.create_scene(
                db,
                s.SceneCreate(name=args.scene, region="synthetic", acquired_before=date(2023, 6, 1), acquired_after=date(2024, 6, 1)),
            )
        added = crud.import_manifest(db, scene.id, root, s.SceneImport(manifest_dir=args.subdir))
        print(f"Scene {scene.id} '{scene.name}': imported {added} tiles")

        if args.predict:
            svc = get_inference()
            tiles = list(db.scalars(select(m.Tile).where(m.Tile.scene_id == scene.id)))
            for t in tiles:
                if not t.predictions:
                    crud.save_prediction(db, t, svc.predict(t.before_path, t.after_path), svc.artifact_dir, commit=False)
            db.commit()
            print(f"Predicted {len(tiles)} tiles with {svc.available()}")


if __name__ == "__main__":
    main()
