#!/usr/bin/env python3
"""
Remove messages older than a given number of days from an mbox-format mailbox.
Preserves original ownership and permissions.

Usage:
    sudo python3 /home/meteor/nmn/bin/clean_mailbox.py /var/spool/mail/ams --days 30
    sudo python3 /home/meteor/nmn/bin/clean_mailbox.py /var/spool/mail/ams --dry-run --days 30
"""

import argparse
import datetime
import os
import shutil
import sys
from mailbox import mbox
from email.utils import parsedate_to_datetime


def main():
    parser = argparse.ArgumentParser(
        description="Remove messages older than N days from an mbox mailbox."
    )
    parser.add_argument("mailbox", help="Path to the mbox file (e.g. /var/spool/mail/ams)")
    parser.add_argument(
        "--days", type=int, default=30,
        help="Keep messages newer than this many days (default: 30)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show how many messages would be removed without changing the file"
    )
    args = parser.parse_args()

    path = args.mailbox
    if not os.path.exists(path):
        print(f"Mailbox not found: {path}")
        sys.exit(0)

    # Record original metadata before rewriting
    st = os.stat(path)
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=args.days)

    mb = mbox(path)
    mb.lock()
    try:
        keys_to_remove = []
        total = 0
        for key in mb.iterkeys():
            total += 1
            msg = mb[key]
            date_header = msg.get("Date")
            remove = False
            if date_header:
                try:
                    dt = parsedate_to_datetime(date_header)
                    if dt < cutoff:
                        remove = True
                except Exception:
                    # Unparseable Date header: keep the message to be safe
                    pass
            # No Date header: keep to be safe
            if remove:
                keys_to_remove.append(key)

        print(f"{path}: {total} messages total, {len(keys_to_remove)} older than {args.days} days")

        if args.dry_run:
            return

        for key in keys_to_remove:
            del mb[key]

        mb.flush()
    finally:
        mb.unlock()
        mb.close()

    # mailbox.mbox writes a new temp file and renames it into place, which can
    # reset ownership/permissions. Restore them.
    if os.path.exists(path):
        os.chmod(path, st.st_mode)
        os.chown(path, st.st_uid, st.st_gid)
        new_st = os.stat(path)
        print(f"Kept {total - len(keys_to_remove)} messages. File size: {st.st_size} -> {new_st.st_size} bytes")
    else:
        print("Warning: mailbox file disappeared after rewrite")


if __name__ == "__main__":
    main()
