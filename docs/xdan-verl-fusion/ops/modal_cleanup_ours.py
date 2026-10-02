"""Terminate Modal sandboxes that belong to a fusion-line run, never anyone else's.

A sandbox is ours only if BOTH hold: its ``instance_id`` tag is a row of the given training parquet(s),
and it was created after the run start. Use --dry-run first. New fusion runs set MODAL_APP_NAME so a
plain app filter suffices; this tool covers runs that used the shared default app "mimoagent".

usage: modal_cleanup_ours.py --data train.parquet [--data ...] --since 2026-10-02T09:41:11Z [--app mimoagent] [--dry-run]
"""

import argparse
import datetime
import json

import modal
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--since", required=True)
    parser.add_argument("--app", default="mimoagent")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ids = set()
    for path in args.data:
        ids.update(str(info["instance_id"]) for info in pd.read_parquet(path)["extra_info"])
    since = datetime.datetime.fromisoformat(args.since.replace("Z", "+00:00"))
    app = modal.App.lookup(args.app, create_if_missing=False)
    matched, skipped = [], 0
    for sandbox in modal.Sandbox.list(app_id=app.app_id):
        tags = sandbox.get_tags()
        created = getattr(sandbox, "created_at", None)
        if tags.get("instance_id") not in ids or (created is not None and created < since):
            skipped += 1
            continue
        matched.append((sandbox.object_id, tags.get("instance_id")))
        if not args.dry_run:
            sandbox.terminate()
    print(json.dumps({"app": args.app, "matched": len(matched), "skipped_not_ours": skipped,
                      "terminated": 0 if args.dry_run else len(matched), "examples": matched[:5]}))


if __name__ == "__main__":
    main()
