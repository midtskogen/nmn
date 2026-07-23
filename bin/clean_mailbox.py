#!/usr/bin/env python3
"""
Remove messages older than a given number of days from an mbox-format mailbox.
Compacts the file in-place, so it works even when the disk is nearly full.
Preserves original ownership and permissions.

Usage:
    sudo python3 /home/meteor/nmn/bin/clean_mailbox.py /var/spool/mail/ams --days 30
    sudo python3 /home/meteor/nmn/bin/clean_mailbox.py /var/spool/mail/ams --dry-run --days 30
"""

import argparse
import datetime
import email
import fcntl
import os
import sys
from email.utils import parsedate_to_datetime


def parse_date_from_bytes(header_bytes):
    """Return datetime for the Date header, or None."""
    try:
        msg = email.message_from_bytes(header_bytes)
        date_hdr = msg.get('Date')
        if date_hdr:
            return parsedate_to_datetime(date_hdr)
    except Exception:
        pass
    return None


def message_older_than(headers_bytes, cutoff):
    """Return True if message Date is parseable and older than cutoff.
    Return False (keep) if Date is missing or unparseable."""
    dt = parse_date_from_bytes(headers_bytes)
    if dt is None:
        return False
    # parsedate_to_datetime may return a naive datetime when the header
    # lacks a timezone. Treat naive dates as UTC so they can be compared
    # with the UTC cutoff.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt < cutoff


def compact_mailbox_in_place(path, cutoff, dry_run=False):
    st = os.stat(path)

    with open(path, 'r+b') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except OSError:
            pass

        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)

        read_pos = 0
        write_pos = 0
        message_count = 0
        removed_count = 0
        kept_count = 0
        bytes_read = 0
        last_report_bytes = 0

        # Skip any preamble before the first "From " line.
        while read_pos < file_size:
            f.seek(read_pos)
            line = f.readline()
            if not line:
                break
            if line.startswith(b'From '):
                break
            bytes_read += len(line)
            read_pos += len(line)

        while read_pos < file_size:
            # Parse the next message.
            msg_start = read_pos
            msg_lines = []
            header_bytes = None
            in_headers = True

            while True:
                f.seek(read_pos)
                line = f.readline()
                if not line:
                    break
                # A new message starts with "From " at the beginning of a line.
                if line.startswith(b'From ') and read_pos > msg_start:
                    break
                msg_lines.append(line)
                if in_headers:
                    if line == b'\n' or line == b'\r\n':
                        in_headers = False
                        header_bytes = b''.join(msg_lines)
                read_pos += len(line)

            message_count += 1
            msg_bytes = b''.join(msg_lines)

            if header_bytes is None:
                header_bytes = msg_bytes
            old = message_older_than(header_bytes, cutoff)

            if old:
                removed_count += 1
            else:
                kept_count += 1
                if not dry_run and msg_bytes and write_pos != msg_start:
                    f.seek(write_pos)
                    f.write(msg_bytes)
                    f.flush()
                write_pos += len(msg_bytes)

            bytes_read += len(msg_bytes)

            # Progress report every ~10% or every 10000 messages
            hit_byte = file_size and (bytes_read - last_report_bytes) >= file_size / 10
            hit_count = message_count > 0 and message_count % 10000 == 0
            if (hit_byte or hit_count) and bytes_read != last_report_bytes:
                pct = bytes_read / file_size * 100 if file_size else 0
                print(f"  Progress: {pct:.1f}% ({bytes_read // (1024*1024)} MB), messages: {message_count}, removed: {removed_count}, kept: {kept_count}", flush=True)
                last_report_bytes = bytes_read

        if dry_run:
            print(f"{path}: {message_count} messages total, {removed_count} older than cutoff, {kept_count} kept (dry run)")
            return

        # Truncate to the new size.
        f.seek(0, 2)
        old_size = f.tell()
        if write_pos < old_size:
            f.truncate(write_pos)
        f.flush()

    # Restore ownership/permissions in case truncate touched them.
    os.chmod(path, st.st_mode)
    try:
        os.chown(path, st.st_uid, st.st_gid)
    except PermissionError:
        pass

    print(f"{path}: {message_count} messages total, {removed_count} removed, {kept_count} kept")
    print(f"Size: {old_size} -> {write_pos} bytes ({write_pos / old_size * 100:.1f}% remaining)")


def main():
    parser = argparse.ArgumentParser(
        description="Remove messages older than N days from an mbox mailbox (in-place, low disk usage)."
    )
    parser.add_argument("mailbox", help="Path to the mbox file (e.g. /var/spool/mail/ams)")
    parser.add_argument(
        "--days", type=int, default=30,
        help="Keep messages newer than this many days (default: 30)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count how many messages would be removed without changing the file"
    )
    args = parser.parse_args()

    path = args.mailbox
    if not os.path.exists(path):
        print(f"Mailbox not found: {path}")
        sys.exit(0)

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=args.days)
    print(f"Cutoff: {cutoff.isoformat()} (keeping messages newer than {args.days} days)")

    compact_mailbox_in_place(path, cutoff, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
