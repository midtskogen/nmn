#!/usr/bin/env python3
"""
Graphical wizard for retraining the NMN meteor classification model.

Guides you through:
  1. Dependency checks
  2. Selecting / splitting positive (meteor) and negative (non-meteor) images
  3. Choosing model-size / performance tradeoffs
  4. Running the training (with live logs and progress)
  5. Evaluating on held-out verification data and installing the best model

Usage:
    python3 nmn/bin/retrain_meteor_model_gui.py
"""

import os
import re
import sys
import json
import time
import shutil
import queue
import random
import pathlib
import fnmatch
import subprocess
import threading
import concurrent.futures
import importlib.util
from datetime import timedelta

# Lazy imports for heavy packages are done inside functions. Only GUI imports are at the top.
try:
    from tkinter import *
    from tkinter import ttk, filedialog, messagebox, scrolledtext
except ImportError as e:
    sys.exit(f"Tkinter is required for this GUI but is not installed: {e}")


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
SCRIPT = pathlib.Path(__file__).resolve()
BIN_DIR = SCRIPT.parent
NMN_DIR = BIN_DIR.parent

CLASSIFY_PY = BIN_DIR / 'classify.py'
MODEL_DIR = NMN_DIR / 'model'

# Default locations for fetching verified / false detections
POS_SOURCE_DEFAULT = NMN_DIR.parent / 'meteor'
NEG_SOURCE_DEFAULT = NMN_DIR.parent / 'wrongs'
FETCH_PATTERN_DEFAULT = 'fireball_orig.jpg'

# tqdm progress bars print a percentage followed by a vertical bar.
_PROGRESS_RE = re.compile(r'\d+%\|')

# -----------------------------------------------------------------------------
# Dependency information
# -----------------------------------------------------------------------------
REQUIRED_DEPS = [
    # (display name, import name, pip package name)
    ('torch', 'torch', 'torch'),
    ('torchvision', 'torchvision', 'torchvision'),
    ('Pillow', 'PIL', 'Pillow'),
    ('OpenCV', 'cv2', 'opencv-python'),
    ('scikit-learn', 'sklearn', 'scikit-learn'),
    ('zstandard', 'zstandard', 'zstandard'),
    ('pandas', 'pandas', 'pandas'),
    ('numpy', 'numpy', 'numpy'),
    ('tqdm', 'tqdm', 'tqdm'),
    ('joblib', 'joblib', 'joblib'),
    ('optuna', 'optuna', 'optuna'),
    ('matplotlib', 'matplotlib', 'matplotlib'),
]

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}


def _dep_installed(import_name):
    try:
        return importlib.util.find_spec(import_name) is not None
    except Exception:
        return False


def human_size(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T']:
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"


def find_images(root):
    """Return a list of image file paths under root."""
    root = pathlib.Path(root)
    files = []
    if root.is_dir():
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                files.append(p)
    return files


def _count_images_dir(path, label, msg_queue=None):
    """Count image files in a flat directory, posting progress every 500 files."""
    count = 0
    if not path.is_dir():
        return 0
    for entry in os.scandir(str(path)):
        if entry.is_file() and entry.name.lower().endswith(tuple(IMAGE_EXTS)):
            count += 1
        if msg_queue and count and count % 500 == 0:
            msg_queue.put(('fetch_status', f"Counting {label}: {count} files..."))
    return count


def link_or_copy(src, dst):
    """Hardlink if possible (Linux), otherwise copy. Overwrites dst."""
    src_p = pathlib.Path(src)
    dst_p = pathlib.Path(dst)
    if dst_p.exists():
        try:
            if dst_p.samefile(src_p):
                return
        except OSError:
            pass
        dst_p.unlink()
    try:
        if sys.platform.startswith('linux'):
            os.link(src_p, dst_p)
        else:
            shutil.copy2(src_p, dst_p)
    except (OSError, AttributeError):
        shutil.copy2(src_p, dst_p)


def _unique_name(dst_dir: pathlib.Path, src: pathlib.Path) -> str:
    """Return a unique destination filename, preserving extension."""
    base = src.stem
    ext = src.suffix
    candidate = base + ext
    counter = 1
    while (dst_dir / candidate).exists():
        candidate = f"{base}_{counter}{ext}"
        counter += 1
    return candidate


def _iter_matching_files(src: pathlib.Path, pattern: str):
    """Yield matching file paths under src, using os.walk for speed."""
    use_glob = '*' in pattern or '?' in pattern or '[' in pattern
    src_str = str(src)
    for root, dirs, files in os.walk(src_str):
        if use_glob:
            matches = fnmatch.filter(files, pattern)
        else:
            matches = [pattern] if pattern in files else []
        for f in matches:
            yield pathlib.Path(root) / f


def _collect_matching_files(src: pathlib.Path, pattern: str, max_workers: int = 16,
                            progress_callback=None):
    """Collect all matching file paths under src in parallel across top-level dirs.

    Over NFS this is much faster than a single-threaded walk because the latency
    of many directory lookups can be overlapped.  If given, progress_callback(done,
    total) is called each time a top-level directory finishes scanning.
    """
    src_str = str(src)
    top = [entry.path for entry in os.scandir(src_str) if entry.is_dir()]
    if not top:
        return list(_iter_matching_files(src, pattern))

    files = []
    done = 0
    total = len(top)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_path = {
            ex.submit(lambda p: list(_iter_matching_files(pathlib.Path(p), pattern)), p): p
            for p in top
        }
        for future in concurrent.futures.as_completed(future_to_path):
            files.extend(future.result())
            done += 1
            if progress_callback:
                progress_callback(done, total)
    return files


def fetch_detection_images(src_root, dst_root, pattern, msg_queue=None, label=''):
    """Efficiently collect image files matching pattern from src_root into dst_root.

    Files are hardlinked when possible; otherwise copied.  The relative path of each
    source file is used as the destination name so identical filenames from different
    events do not collide.  Progress messages are posted to msg_queue every 50 files.
    """
    src = pathlib.Path(src_root)
    dst = pathlib.Path(dst_root)
    if not src.is_dir():
        raise ValueError(f"Source directory does not exist: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    # Clear any stale files from previous fetches so the destination is deterministic.
    for existing in dst.iterdir():
        if existing.is_file():
            existing.unlink()

    def scan_progress(done, total):
        if msg_queue and (done % 10 == 0 or done == total):
            msg_queue.put(('fetch_status', f"Scanning {label}: {done}/{total} subdirs..."))

    if msg_queue:
        msg_queue.put(('fetch_status', f"Scanning {label} (parallel)..."))

    files = _collect_matching_files(src, pattern, progress_callback=scan_progress)

    if msg_queue:
        msg_queue.put(('fetch_status', f"Fetching {label}: copying {len(files)} files..."))

    copied = 0
    update_every = 50
    for p in files:
        if not p.is_file():
            continue
        name = '_'.join(p.relative_to(src).parts)
        if (dst / name).exists():
            name = _unique_name(dst, p)
        link_or_copy(p, dst / name)
        copied += 1
        if msg_queue and copied % update_every == 0:
            msg_queue.put(('fetch_progress', (copied, 0)))
            msg_queue.put(('fetch_status', f"Fetching {label}: {copied}/{len(files)} files copied..."))
    return copied


def compute_image_stats(image_paths, max_sample=200):
    """Count files and (optionally) sample dimensions."""
    ext_counts = {}
    for p in image_paths:
        ext = p.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    dims = []
    sample = image_paths if len(image_paths) <= max_sample else random.sample(image_paths, max_sample)
    for p in sample:
        try:
            from PIL import Image
            with Image.open(p) as img:
                dims.append(img.size)
        except Exception:
            pass

    if dims:
        mw = sum(d[0] for d in dims) / len(dims)
        mh = sum(d[1] for d in dims) / len(dims)
    else:
        mw = mh = 0
    return {
        'count': len(image_paths),
        'ext_counts': ext_counts,
        'sampled': len(dims),
        'mean_width': mw,
        'mean_height': mh,
    }


def split_datasets(pos_dir, neg_dir, work_dir, verify_ratio, msg_queue=None, pos_files=None, neg_files=None):
    """Split positive/negative images into train and held-out verification sets.

    If pos_files/neg_files are already known they can be passed in to avoid
    re-scanning.  Copy progress is reported to msg_queue every 50 files.
    """
    pos_files = pos_files or find_images(pos_dir)
    neg_files = neg_files or find_images(neg_dir)
    if not pos_files:
        raise ValueError(f"No images found in positive directory: {pos_dir}")
    if not neg_files:
        raise ValueError(f"No images found in negative directory: {neg_dir}")

    all_files = pos_files + neg_files
    labels = [1] * len(pos_files) + [0] * len(neg_files)

    try:
        from sklearn.model_selection import train_test_split
        train_files, verify_files, train_labels, verify_labels = train_test_split(
            all_files, labels, test_size=verify_ratio, stratify=labels, random_state=42
        )
    except Exception as exc:
        raise RuntimeError(f"Could not split data: {exc}")

    dirs = {
        'train/meteor': [p for p, l in zip(train_files, train_labels) if l == 1],
        'train/non_meteor': [p for p, l in zip(train_files, train_labels) if l == 0],
        'verify/meteor': [p for p, l in zip(verify_files, verify_labels) if l == 1],
        'verify/non_meteor': [p for p, l in zip(verify_files, verify_labels) if l == 0],
    }

    work = pathlib.Path(work_dir)
    total_files = sum(len(files) for files in dirs.values())
    copied = 0
    for name, files in dirs.items():
        d = work / name
        d.mkdir(parents=True, exist_ok=True)
        if msg_queue:
            msg_queue.put(('prepare_status', f"Copying {len(files)} files to {name}..."))
        for f in files:
            link_or_copy(f, d / f.name)
            copied += 1
            if msg_queue and copied % 50 == 0:
                msg_queue.put(('prepare_copy_progress', (copied, total_files)))

    return {
        'train_pos': len(dirs['train/meteor']),
        'train_neg': len(dirs['train/non_meteor']),
        'verify_pos': len(dirs['verify/meteor']),
        'verify_neg': len(dirs['verify/non_meteor']),
        'work_dir': work,
    }


def parse_evaluate_output(text):
    """Parse the printed metrics from classify.py evaluate."""
    result = {}
    for line in text.splitlines():
        m = re.search(r'Optimal Classification Threshold:\s*([\d.]+)', line)
        if m:
            result['threshold'] = float(m.group(1))
        m = re.search(r'F1-Score:\s*([\d.]+)', line)
        if m:
            result['f1'] = float(m.group(1))
        m = re.search(r'Precision:\s*([\d.]+)', line)
        if m:
            result['precision'] = float(m.group(1))
        m = re.search(r'Recall:\s*([\d.]+)', line)
        if m:
            result['recall'] = float(m.group(1))
    return result


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------
class RetrainApp(Tk):
    def __init__(self):
        super().__init__()
        self.title("NMN meteor model retraining wizard")
        self.geometry("1000x700")
        self.minsize(850, 600)

        self.steps = [
            "1. Dependencies",
            "2. Data & split",
            "3. Training options",
            "4. Train",
            "5. Results",
        ]
        self.current_step = 0
        self.results = []  # list of dicts for each cluster count
        self.progress_log_line = None  # line number of the active tqdm/progress line
        self.deps_ok = False
        self.split_info = None

        # Thread / subprocess control
        self.worker_thread = None
        self.current_proc = None
        self.stop_requested = False
        self.msg_queue = queue.Queue()

        self.build_ui()
        self.show_step(0)
        self.after(100, self.process_queue)
        self.check_dependencies()

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------
    def build_ui(self):
        # Top info
        header = ttk.Frame(self, padding=10)
        header.pack(fill=X)
        ttk.Label(header, text="Retrain meteor_efficientnet_b0_model_clustered.pth.zst",
                  font=('Helvetica', 14, 'bold')).pack(anchor=W)
        ttk.Label(header, text=(
            "This wizard guides you through preparing data, training EfficientNet-B0, "
            "and evaluating on held-out verification images."
        )).pack(anchor=W, pady=(2, 0))

        # Bottom navigation: pack before the expanding main frame so it is
        # never pushed below the visible window when the content is tall.
        footer = ttk.Frame(self, padding=10)
        footer.pack(fill=X, side=BOTTOM)

        self.back_btn = ttk.Button(footer, text="Back", command=self.go_back)
        self.back_btn.pack(side=LEFT)
        self.next_btn = ttk.Button(footer, text="Next", command=self.go_next)
        self.next_btn.pack(side=RIGHT)

        # Sidebar + content
        main = ttk.Frame(self)
        main.pack(fill=BOTH, expand=True, padx=10, pady=5)

        self.sidebar = ttk.Frame(main, width=150)
        self.sidebar.pack(side=LEFT, fill=Y)
        self.sidebar.pack_propagate(False)

        self.step_labels = []
        for i, name in enumerate(self.steps):
            lbl = ttk.Label(self.sidebar, text=name, padding=8,
                            font=('Helvetica', 10, 'bold'))
            lbl.pack(fill=X, pady=2)
            self.step_labels.append(lbl)

        self.content = ttk.Frame(main, padding=10)
        self.content.pack(side=LEFT, fill=BOTH, expand=True)

        self.build_pages()

    def build_pages(self):
        self.pages = []

        # ---- Page 0: Dependencies ----
        p0 = ttk.Frame(self.content)
        ttk.Label(p0, text="Prerequisite checks", font=('Helvetica', 12, 'bold')).pack(anchor=W, pady=(0, 10))

        self.dep_status = ttk.Frame(p0)
        self.dep_status.pack(fill=X, expand=True)

        self.dep_cmd = Text(p0, height=3, wrap=WORD, state='disabled', bg='#f5f5f5')
        self.dep_cmd.pack(fill=X, pady=10)
        self.set_text(self.dep_cmd, "Install command:\nRun a check to see missing packages.")

        ttk.Button(p0, text="Refresh checks", command=self.check_dependencies).pack(anchor=W)
        self.pages.append(p0)

        # ---- Page 1: Data ----
        p1 = ttk.Frame(self.content)
        ttk.Label(p1, text="Training data", font=('Helvetica', 12, 'bold')).pack(anchor=W, pady=(0, 10))

        work_default = NMN_DIR.parent / 'retrain_work'

        # Positive dir
        f1 = ttk.Frame(p1)
        f1.pack(fill=X, pady=5)
        ttk.Label(f1, text="Positive directory (meteors):").pack(anchor=W)
        self.pos_dir = StringVar(value=str(work_default / 'positive_fetched'))
        e1 = ttk.Entry(f1, textvariable=self.pos_dir)
        e1.pack(side=LEFT, fill=X, expand=True)
        ttk.Button(f1, text="Browse...", command=lambda: self.browse_dir(self.pos_dir)).pack(side=LEFT, padx=5)

        # Negative dir
        f2 = ttk.Frame(p1)
        f2.pack(fill=X, pady=5)
        ttk.Label(f2, text="Negative directory (non-meteors / false detections):").pack(anchor=W)
        self.neg_dir = StringVar(value=str(work_default / 'negative_fetched'))
        e2 = ttk.Entry(f2, textvariable=self.neg_dir)
        e2.pack(side=LEFT, fill=X, expand=True)
        ttk.Button(f2, text="Browse...", command=lambda: self.browse_dir(self.neg_dir)).pack(side=LEFT, padx=5)

        # Work dir
        f3 = ttk.Frame(p1)
        f3.pack(fill=X, pady=5)
        ttk.Label(f3, text="Working directory (train/verify split will be created here):").pack(anchor=W)
        self.work_dir = StringVar(value=str(NMN_DIR.parent / 'retrain_work'))
        e3 = ttk.Entry(f3, textvariable=self.work_dir)
        e3.pack(side=LEFT, fill=X, expand=True)
        ttk.Button(f3, text="Browse...", command=lambda: self.browse_dir(self.work_dir, create=True)).pack(side=LEFT, padx=5)

        # Split
        f4 = ttk.Frame(p1)
        f4.pack(fill=X, pady=10)
        ttk.Label(f4, text="Verification split ratio:").pack(side=LEFT)
        self.split_ratio = DoubleVar(value=0.20)
        ttk.Spinbox(f4, from_=0.05, to=0.50, increment=0.05, textvariable=self.split_ratio, width=5).pack(side=LEFT, padx=5)
        ttk.Label(f4, text="(fraction kept unseen for final evaluation)").pack(side=LEFT)

        prepare_btn_frame = ttk.Frame(p1)
        prepare_btn_frame.pack(anchor=W, pady=10)
        self.prepare_btn = ttk.Button(prepare_btn_frame, text="Scan & prepare split", command=self.start_prepare)
        self.prepare_btn.pack(side=LEFT)
        self.check_scanned_btn = ttk.Button(prepare_btn_frame, text="Check scanned", command=self.check_scanned)
        self.check_scanned_btn.pack(side=LEFT, padx=(10, 0))
        self.prepare_status = ttk.Label(prepare_btn_frame, text="", wraplength=500, justify=LEFT)
        self.prepare_status.pack(side=LEFT, padx=(10, 0))

        self.prepare_progress = ttk.Progressbar(p1, orient=HORIZONTAL, mode='indeterminate')
        self.prepare_progress.pack(fill=X, pady=(5, 0))

        # ----------------------------------------------------------------------
        # Fetch detections from report directories
        # ----------------------------------------------------------------------
        fetch_frame = ttk.LabelFrame(p1, text="Fetch detections from report directories", padding=10)
        fetch_frame.pack(fill=X, pady=15)

        self.pos_source = StringVar(value=str(POS_SOURCE_DEFAULT) if POS_SOURCE_DEFAULT.is_dir() else '')
        self.neg_source = StringVar(value=str(NEG_SOURCE_DEFAULT) if NEG_SOURCE_DEFAULT.is_dir() else '')
        self.fetch_pattern = StringVar(value=FETCH_PATTERN_DEFAULT)
        self.fetch_pos_out = StringVar(value=str(pathlib.Path(self.work_dir.get()) / 'positive_fetched'))
        self.fetch_neg_out = StringVar(value=str(pathlib.Path(self.work_dir.get()) / 'negative_fetched'))

        def make_fetch_row(parent, label, var, create=False, browse=True):
            ff = ttk.Frame(parent)
            ff.pack(fill=X, pady=2)
            ttk.Label(ff, text=label, width=26).pack(side=LEFT)
            ent = ttk.Entry(ff, textvariable=var)
            ent.pack(side=LEFT, fill=X, expand=True, padx=(5, 0))
            if browse:
                ttk.Button(ff, text="Browse...", command=lambda: self.browse_dir(var, create=create)).pack(side=LEFT, padx=5)

        make_fetch_row(fetch_frame, "Verified meteor reports:", self.pos_source)
        make_fetch_row(fetch_frame, "False detections (wrongs):", self.neg_source)
        make_fetch_row(fetch_frame, "File pattern to collect:", self.fetch_pattern, browse=False)
        make_fetch_row(fetch_frame, "Output positive dir:", self.fetch_pos_out, create=True)
        make_fetch_row(fetch_frame, "Output negative dir:", self.fetch_neg_out, create=True)

        fetch_btn_frame = ttk.Frame(fetch_frame)
        fetch_btn_frame.pack(fill=X, pady=(10, 0))
        self.fetch_btn = ttk.Button(fetch_btn_frame, text="Fetch detections", command=self.start_fetch)
        self.fetch_btn.pack(side=LEFT)
        self.scan_counts_btn = ttk.Button(fetch_btn_frame, text="Count fetched", command=self.count_sources)
        self.scan_counts_btn.pack(side=LEFT, padx=(10, 0))
        self.fetch_status = ttk.Label(fetch_btn_frame, text="Ready to fetch.", wraplength=700, justify=LEFT)
        self.fetch_status.pack(side=LEFT, padx=(10, 0))

        self.fetch_progress = ttk.Progressbar(fetch_frame, orient=HORIZONTAL, mode='indeterminate')
        self.fetch_progress.pack(fill=X, pady=(5, 0))

        self.source_count_status = ttk.Label(fetch_frame, text="", wraplength=700, justify=LEFT)
        self.source_count_status.pack(anchor=W, pady=(5, 0))

        self.data_stats = scrolledtext.ScrolledText(p1, height=5, wrap=WORD, state='disabled', bg='#f5f5f5')
        self.data_stats.pack(fill=BOTH, expand=True, pady=10)
        self.set_text(self.data_stats, "No data scanned yet.")
        self.pages.append(p1)

        # ---- Page 2: Options ----
        p2 = ttk.Frame(self.content)
        ttk.Label(p2, text="Training options", font=('Helvetica', 12, 'bold')).pack(anchor=W, pady=(0, 10))

        # Output dir
        fo = ttk.Frame(p2)
        fo.pack(fill=X, pady=5)
        ttk.Label(fo, text="Output directory for new models:").pack(anchor=W)
        self.output_dir = StringVar(value=str(NMN_DIR / 'retrain_output'))
        eout = ttk.Entry(fo, textvariable=self.output_dir)
        eout.pack(side=LEFT, fill=X, expand=True)
        ttk.Button(fo, text="Browse...", command=lambda: self.browse_dir(self.output_dir, create=True)).pack(side=LEFT, padx=5)

        # Epochs / batch
        fo2 = ttk.Frame(p2)
        fo2.pack(fill=X, pady=10)
        ttk.Label(fo2, text="Epochs:").pack(side=LEFT)
        self.epochs = IntVar(value=50)
        ttk.Spinbox(fo2, from_=1, to=500, textvariable=self.epochs, width=6).pack(side=LEFT, padx=5)

        ttk.Label(fo2, text="Batch size:").pack(side=LEFT, padx=(15, 0))
        self.batch_size = IntVar(value=16)
        ttk.Spinbox(fo2, from_=1, to=256, textvariable=self.batch_size, width=5).pack(side=LEFT, padx=5)

        # Model size / performance
        ttk.Label(p2, text="K-Means weight clustering (model size vs accuracy tradeoff):",
                  font=('Helvetica', 10, 'bold')).pack(anchor=W, pady=(10, 0))

        self.cluster_frame = ttk.Frame(p2)
        self.cluster_frame.pack(fill=X, pady=5)

        ttk.Label(self.cluster_frame,
                  text="Select one cluster count to evaluate. "
                       "Lower K = smaller model, lower fidelity. Higher K = larger model, closer to original.",
                  wraplength=800).pack(anchor=W)

        self.cluster_listbox = Listbox(self.cluster_frame, selectmode=SINGLE, height=6, exportselection=False)
        for k in [64, 128, 256, 512]:
            self.cluster_listbox.insert(END, str(k))
        self.cluster_listbox.pack(anchor=W, pady=5)
        # Default select 128
        self.cluster_listbox.selection_set(1)

        # Options checkboxes
        self.balance = BooleanVar(value=True)
        self.scheduler = BooleanVar(value=True)
        ttk.Checkbutton(p2, text="Balance classes by generating synthetic negatives", variable=self.balance).pack(anchor=W, pady=2)
        ttk.Checkbutton(p2, text="Use learning-rate scheduler (ReduceLROnPlateau)", variable=self.scheduler).pack(anchor=W, pady=2)

        ttk.Label(p2, text="Input size is fixed at 192x96 to match nmn/bin/predict.py.",
                  foreground='gray').pack(anchor=W, pady=10)

        self.pages.append(p2)

        # ---- Page 3: Train ----
        p3 = ttk.Frame(self.content)
        ttk.Label(p3, text="Training progress", font=('Helvetica', 12, 'bold')).pack(anchor=W, pady=(0, 10))

        self.train_status = ttk.Label(p3, text="Ready to start.")
        self.train_status.pack(anchor=W)

        self.progress = ttk.Progressbar(p3, orient=HORIZONTAL, mode='determinate')
        self.progress.pack(fill=X, pady=10)

        btn_frame = ttk.Frame(p3)
        btn_frame.pack(fill=X, pady=5)
        self.start_train_btn = ttk.Button(btn_frame, text="Start training", command=self.start_training)
        self.start_train_btn.pack(side=LEFT, padx=5)
        self.stop_train_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_training, state=DISABLED)
        self.stop_train_btn.pack(side=LEFT, padx=5)

        ttk.Label(p3, text="Log output:").pack(anchor=W, pady=(10, 0))
        self.log_box = scrolledtext.ScrolledText(p3, height=12, state='disabled', wrap=WORD)
        self.log_box.pack(fill=BOTH, expand=True, pady=5)

        self.pages.append(p3)

        # ---- Page 4: Results ----
        p4 = ttk.Frame(self.content)
        ttk.Label(p4, text="Results", font=('Helvetica', 12, 'bold')).pack(anchor=W, pady=(0, 10))

        self.results_summary = ttk.Label(p4, text="No results yet.", wraplength=850, justify=LEFT)
        self.results_summary.pack(anchor=W)

        cols = ('K', 'Size', 'F1', 'Precision', 'Recall', 'Threshold')
        self.tree = ttk.Treeview(p4, columns=cols, show='headings', height=8)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=90, anchor='center')
        self.tree.pack(fill=BOTH, expand=True, pady=10)
        self.tree.bind('<<TreeviewSelect>>', self.on_result_select)

        rbtn = ttk.Frame(p4)
        rbtn.pack(fill=X)
        ttk.Button(rbtn, text="Replace current model with selected", command=self.install_model).pack(side=LEFT, padx=5)
        ttk.Button(rbtn, text="Save results to JSON", command=self.save_results_json).pack(side=LEFT, padx=5)

        self.pages.append(p4)

    # -------------------------------------------------------------------------
    # Page navigation
    # -------------------------------------------------------------------------
    def show_step(self, idx):
        for i, lbl in enumerate(self.step_labels):
            if i == idx:
                lbl.configure(background='#0078d7', foreground='white')
            else:
                lbl.configure(background='', foreground='')
        for p in self.pages:
            p.pack_forget()
        self.pages[idx].pack(fill=BOTH, expand=True)
        self.current_step = idx
        self.update_navigation()

    def update_navigation(self):
        self.back_btn.configure(state=NORMAL if self.current_step > 0 else DISABLED)
        can_advance = self.current_step < 3
        if self.current_step == 0:
            can_advance = self.deps_ok
        elif self.current_step == 1:
            can_advance = self.split_info is not None
        self.next_btn.configure(text="Next", state=NORMAL if can_advance else DISABLED)

    def go_back(self):
        if self.current_step > 0:
            self.show_step(self.current_step - 1)

    def go_next(self):
        if self.current_step < len(self.steps) - 1:
            self.show_step(self.current_step + 1)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def set_text(self, widget, text):
        widget.configure(state='normal')
        widget.delete('1.0', END)
        widget.insert('1.0', text)
        widget.configure(state='disabled')

    def log(self, line, overwrite=False):
        """Append a line to the log. Progress lines overwrite the active
        progress line instead of creating a new one."""
        is_progress = overwrite or bool(_PROGRESS_RE.search(line))
        self.log_box.configure(state='normal')
        if is_progress:
            # Remove the previous progress line and rewrite it at the end.
            if self.progress_log_line is not None:
                try:
                    self.log_box.delete(
                        f"{self.progress_log_line}.0",
                        f"{self.progress_log_line + 1}.0",
                    )
                except Exception:
                    pass
            self.log_box.insert(END, line + '\n')
            # The Text widget always keeps a trailing empty line after a \n,
            # so the line we just wrote is two lines above 'end'.
            end_line = int(self.log_box.index('end').split('.')[0])
            self.progress_log_line = end_line - 2
        else:
            self.progress_log_line = None
            self.log_box.insert(END, line + '\n')
        self.log_box.see(END)
        self.log_box.configure(state='disabled')

    def browse_dir(self, var, create=False):
        d = filedialog.askdirectory(initialdir=var.get() or str(NMN_DIR))
        if d:
            p = pathlib.Path(d)
            if create:
                p.mkdir(parents=True, exist_ok=True)
            var.set(str(p))

    def check_dependencies(self):
        for w in self.dep_status.winfo_children():
            w.destroy()

        all_ok = True
        missing_pip = []
        for name, import_name, pip_name in REQUIRED_DEPS:
            ok = _dep_installed(import_name)
            if not ok:
                all_ok = False
                missing_pip.append(pip_name)
            color = 'green' if ok else 'red'
            text = f"{name}: {'OK' if ok else 'MISSING'}"
            lbl = ttk.Label(self.dep_status, text=text, foreground=color, font=('Courier', 10))
            lbl.pack(anchor=W, pady=1)

        classify_ok = CLASSIFY_PY.is_file()
        if not classify_ok:
            all_ok = False
        color = 'green' if classify_ok else 'red'
        lbl = ttk.Label(self.dep_status, text=f"{CLASSIFY_PY}: {'OK' if classify_ok else 'NOT FOUND'}",
                        foreground=color, font=('Courier', 10))
        lbl.pack(anchor=W, pady=1)

        if missing_pip:
            cmd = "python3 -m pip install " + " ".join(missing_pip)
            self.set_text(self.dep_cmd, f"Install command:\n{cmd}")
            header_text = "Some dependencies are missing. Install them before continuing."
            header_color = 'red'
        else:
            self.set_text(self.dep_cmd, "All required packages are installed.")
            header_text = "All dependencies found. You can continue."
            header_color = 'green'

        self.dep_status_header = ttk.Label(self.dep_status, text=header_text,
                                          foreground=header_color, font=('Helvetica', 10, 'bold'))
        self.dep_status_header.pack(anchor=W, pady=(10, 0))

        self.deps_ok = all_ok
        self.update_navigation()

    def check_scanned(self):
        pos = self.pos_dir.get().strip()
        neg = self.neg_dir.get().strip()
        work = self.work_dir.get().strip()

        pos_path = pathlib.Path(pos) if pos else None
        neg_path = pathlib.Path(neg) if neg else None
        work_p = pathlib.Path(work) if work else None

        pos_count = _count_images_dir(pos_path, 'positives') if pos_path and pos_path.is_dir() else 0
        neg_count = _count_images_dir(neg_path, 'negatives') if neg_path and neg_path.is_dir() else 0

        status = f"Scanned: {pos_count} positives, {neg_count} negatives"

        # If a previous train/verify split exists under the work directory,
        # allow training to proceed without re-running 'Scan & prepare split'.
        if work_p and work_p.is_dir():
            train_pos = _count_images_dir(work_p / 'train' / 'meteor', 'train positives')
            train_neg = _count_images_dir(work_p / 'train' / 'non_meteor', 'train negatives')
            verify_pos = _count_images_dir(work_p / 'verify' / 'meteor', 'verify positives')
            verify_neg = _count_images_dir(work_p / 'verify' / 'non_meteor', 'verify negatives')

            if all((train_pos, train_neg, verify_pos, verify_neg)):
                self.split_info = {
                    'train_pos': train_pos,
                    'train_neg': train_neg,
                    'verify_pos': verify_pos,
                    'verify_neg': verify_neg,
                    'work_dir': work_p,
                }
                summary = (
                    f"Reusing existing split under: {work_p}\n"
                    f"  Training:     {train_pos} meteors, {train_neg} non-meteors\n"
                    f"  Verification: {verify_pos} meteors, {verify_neg} non-meteors\n"
                )
                if train_pos < 10 or train_neg < 10:
                    summary += "WARNING: Very small training set.\n"
                self.set_text(self.data_stats, summary)
                status += ". Split found; ready to train."
            else:
                self.split_info = None
                status += " (no complete existing split; use Scan & prepare split)"
        else:
            self.split_info = None
            status += " (no work directory set)"

        self.prepare_status.configure(text=status)
        self.update_navigation()

    def start_prepare(self):
        pos = self.pos_dir.get().strip()
        neg = self.neg_dir.get().strip()
        work = self.work_dir.get().strip()
        if not pos or not pathlib.Path(pos).is_dir():
            messagebox.showerror("Error", "Please select a valid positive (meteor) directory.")
            return
        if not neg or not pathlib.Path(neg).is_dir():
            messagebox.showerror("Error", "Please select a valid negative (non-meteor) directory.")
            return

        try:
            ratio = float(self.split_ratio.get())
            if not 0 < ratio < 1:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Error", "Verification split ratio must be between 0 and 1.")
            return

        self.split_info = None
        self.update_navigation()
        self.prepare_btn.configure(state=DISABLED)
        self.prepare_progress.stop()
        self.prepare_progress.configure(mode='determinate', maximum=100, value=0)
        self.prepare_status.configure(text="Preparing data...")

        t = threading.Thread(
            target=self.prepare_worker,
            args=(pos, neg, work, ratio),
            daemon=True,
        )
        t.start()

    def prepare_worker(self, pos, neg, work, ratio):
        try:
            work_p = pathlib.Path(work)
            if work_p.exists():
                self.msg_queue.put(('prepare_status', 'Cleaning previous train/verify split...'))
                try:
                    shutil.rmtree(work_p / 'train', ignore_errors=True)
                    shutil.rmtree(work_p / 'verify', ignore_errors=True)
                except Exception:
                    pass

            self.msg_queue.put(('prepare_status', 'Scanning positive images...'))
            self.msg_queue.put(('prepare_progress', (0, 100)))
            pos_files = find_images(pos)

            self.msg_queue.put(('prepare_status', 'Scanning negative images...'))
            self.msg_queue.put(('prepare_progress', (20, 100)))
            neg_files = find_images(neg)

            if len(pos_files) > len(neg_files) and len(neg_files) >= 2000:
                pos_files = random.Random(42).sample(pos_files, len(neg_files))
                self.msg_queue.put(('prepare_status', f"Undersampled positives to {len(pos_files)} to balance classes"))

            pos_stats = compute_image_stats(pos_files)
            neg_stats = compute_image_stats(neg_files)

            summary = (
                f"Positive class: {pos_stats['count']} images {pos_stats['ext_counts']}\n"
                f"Negative class: {neg_stats['count']} images {neg_stats['ext_counts']}\n"
                f"Mean dimensions (sample): {pos_stats['mean_width']:.0f}x{pos_stats['mean_height']:.0f} (positive), "
                f"{neg_stats['mean_width']:.0f}x{neg_stats['mean_height']:.0f} (negative)\n\n"
            )

            self.msg_queue.put(('prepare_status', 'Splitting dataset...'))
            self.msg_queue.put(('prepare_progress', (40, 100)))
            split_info = split_datasets(pos, neg, work, ratio, self.msg_queue, pos_files=pos_files, neg_files=neg_files)
            summary += (
                f"Split created under: {work_p}\n"
                f"  Training:   {split_info['train_pos']} meteors, {split_info['train_neg']} non-meteors\n"
                f"  Verification: {split_info['verify_pos']} meteors, {split_info['verify_neg']} non-meteors\n\n"
            )
            if split_info['train_pos'] < 10 or split_info['train_neg'] < 10:
                summary += "WARNING: Very small training set. Consider collecting more samples.\n"
            if abs(split_info['train_pos'] - split_info['train_neg']) / max(split_info['train_pos'] + split_info['train_neg'], 1) > 0.7:
                summary += "NOTE: Classes are imbalanced. Enable 'Balance classes' or collect more of the minority class.\n"

            self.msg_queue.put(('prepare_progress', (100, 100)))
            self.msg_queue.put(('prepare_done', {'summary': summary, 'split_info': split_info, 'error': None}))
        except Exception as exc:
            self.msg_queue.put(('prepare_done', {'summary': '', 'split_info': None, 'error': str(exc)}))

    def start_fetch(self):
        pos_src = self.pos_source.get().strip()
        neg_src = self.neg_source.get().strip()
        pattern = self.fetch_pattern.get().strip() or FETCH_PATTERN_DEFAULT
        pos_out = self.fetch_pos_out.get().strip()
        neg_out = self.fetch_neg_out.get().strip()

        if not pos_src or not pathlib.Path(pos_src).is_dir():
            messagebox.showerror("Error", "Please select a valid verified meteor report directory.")
            return
        if not neg_src or not pathlib.Path(neg_src).is_dir():
            messagebox.showerror("Error", "Please select a valid false detections directory.")
            return

        self.fetch_btn.configure(state=DISABLED)
        self.fetch_progress.configure(mode='indeterminate')
        self.fetch_progress.start()
        t = threading.Thread(
            target=self.fetch_worker,
            args=(pos_src, neg_src, pattern, pos_out, neg_out),
            daemon=True,
        )
        t.start()

    def fetch_worker(self, pos_src, neg_src, pattern, pos_out, neg_out):
        try:
            pos_count = fetch_detection_images(pos_src, pos_out, pattern, self.msg_queue, label='positives')
            neg_count = fetch_detection_images(neg_src, neg_out, pattern, self.msg_queue, label='negatives')

            summary = f"Fetched {pos_count} positives and {neg_count} negatives."
            self.msg_queue.put(('fetch_status', summary))
            self.msg_queue.put(('fetch_done', (pos_out, neg_out)))
        except Exception as exc:
            self.msg_queue.put(('fetch_status', f"Fetch failed: {exc}"))
            self.msg_queue.put(('fetch_done', None))

    def count_sources(self):
        pos_out = self.fetch_pos_out.get().strip()
        neg_out = self.fetch_neg_out.get().strip()

        self.fetch_btn.configure(state=DISABLED)
        self.scan_counts_btn.configure(state=DISABLED)
        self.fetch_progress.configure(mode='indeterminate')
        self.fetch_progress.start()
        self.fetch_status.configure(text="Counting already-fetched images...")

        t = threading.Thread(
            target=self.count_worker,
            args=(pos_out, neg_out),
            daemon=True,
        )
        t.start()

    def count_worker(self, pos_out, neg_out):
        try:
            pos_path = pathlib.Path(pos_out)
            neg_path = pathlib.Path(neg_out)
            pos_count = _count_images_dir(pos_path, 'positives', self.msg_queue) if pos_path.is_dir() else 0
            neg_count = _count_images_dir(neg_path, 'negatives', self.msg_queue) if neg_path.is_dir() else 0
            self.msg_queue.put(('source_counts', (pos_count, neg_count)))
            self.msg_queue.put(('fetch_status', f"Found {pos_count} positives and {neg_count} negatives already fetched"))
        except Exception as exc:
            self.msg_queue.put(('fetch_status', f"Count failed: {exc}"))
        finally:
            self.msg_queue.put(('fetch_done', None))

    def selected_cluster_counts(self):
        sel = self.cluster_listbox.curselection()
        if not sel:
            return [256]
        return sorted(int(self.cluster_listbox.get(i)) for i in sel)

    def start_training(self):
        if not getattr(self, 'deps_ok', False):
            messagebox.showerror("Error", "Dependencies are not satisfied. Install them first.")
            return

        if not getattr(self, 'split_info', None):
            messagebox.showerror("Error", "Please prepare the train/verify split on the Data page first.")
            return

        output = pathlib.Path(self.output_dir.get())
        output.mkdir(parents=True, exist_ok=True)

        clusters = self.selected_cluster_counts()
        if not clusters:
            messagebox.showerror("Error", "Please select at least one cluster count.")
            return

        self.results = []
        self.progress_log_line = None
        self.stop_requested = False
        self.progress['value'] = 0
        self.set_text(self.log_box, "")
        self.start_train_btn.configure(state=DISABLED)
        self.stop_train_btn.configure(state=NORMAL)
        self.show_step(3)

        t = threading.Thread(
            target=self.training_worker,
            args=(output, clusters),
            daemon=True
        )
        t.start()
        self.worker_thread = t

    def stop_training(self):
        self.stop_requested = True
        if self.current_proc:
            try:
                self.current_proc.terminate()
            except Exception:
                pass
        self.train_status.configure(text="Stopping...")

    # -------------------------------------------------------------------------
    # Training pipeline (runs in a worker thread)
    # -------------------------------------------------------------------------
    def training_worker(self, output_dir, clusters):
        start_time = time.time()

        train_pos = self.split_info['work_dir'] / 'train' / 'meteor'
        train_neg = self.split_info['work_dir'] / 'train' / 'non_meteor'
        verify_pos = self.split_info['work_dir'] / 'verify' / 'meteor'
        verify_neg = self.split_info['work_dir'] / 'verify' / 'non_meteor'

        current_model = MODEL_DIR / 'meteor_efficientnet_b0_model_clustered.pth.zst'
        evaluate_current = current_model.is_file()
        total_jobs = 1 + len(clusters) + (1 if evaluate_current else 0)
        done_jobs = 0

        # ---- Base training (unclustered) ----
        self.msg_queue.put(('status', "Training base EfficientNet-B0 model..."))
        base_cmd = [
            sys.executable, str(CLASSIFY_PY), '-v', 'efficientnet',
            '--epochs', str(self.epochs.get()),
            '--batch-size', str(self.batch_size.get()),
            '--img-width', '192', '--img-height', '96',
        ]
        if self.balance.get():
            base_cmd.append('--balance')
        if self.scheduler.get():
            base_cmd.append('--scheduler')
        base_cmd += [str(train_pos), str(train_neg)]

        rc = self.run_command(base_cmd, cwd=str(output_dir))
        if rc != 0 or self.stop_requested:
            self.msg_queue.put(('status', "Training failed."))
            self.msg_queue.put(('finished', None))
            return

        done_jobs += 1
        self.msg_queue.put(('progress', int(100 * done_jobs / total_jobs)))

        base_pth = output_dir / 'meteor_efficientnet_b0_model.pth'
        if not base_pth.exists():
            self.msg_queue.put(('status', "Base model file not found after training."))
            self.msg_queue.put(('finished', None))
            return

        # ---- Evaluate current live model as a baseline ----
        if evaluate_current and not self.stop_requested:
            self.msg_queue.put(('status', "Evaluating current live model as baseline..."))
            eval_cmd = [
                sys.executable, str(CLASSIFY_PY), '-v', 'evaluate',
                '-m', str(current_model),
                str(verify_pos), str(verify_neg),
                '--img-width', '192', '--img-height', '96',
                '--batch-size', str(self.batch_size.get()),
            ]
            eval_stdout, rc = self.run_command_with_output(eval_cmd, cwd=str(output_dir))
            metrics = parse_evaluate_output(eval_stdout) if rc == 0 else {}
            self.results.append({
                'k': 'current',
                'path': str(current_model),
                'size': current_model.stat().st_size,
                'metrics': metrics,
            })
            self.msg_queue.put(('result', self.results[-1]))
            done_jobs += 1
            self.msg_queue.put(('progress', int(100 * done_jobs / total_jobs)))

        # ---- Cluster sweep ----
        for k in clusters:
            if self.stop_requested:
                break

            self.msg_queue.put(('status', f"Clustering with K={k}..."))
            k_dir = output_dir / f'k{k}'
            k_dir.mkdir(parents=True, exist_ok=True)

            # Hardlink/copy base model into the per-K directory
            link_or_copy(base_pth, k_dir / 'meteor_efficientnet_b0_model.pth')

            # Run clustering using classify._run_clustering
            cluster_code = (
                "import logging, sys; "
                "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'); "
                f"sys.path.insert(0, {str(BIN_DIR)!r}); "
                "import argparse, classify; "
                f"args = argparse.Namespace(model_name='efficientnet_b0', clusters={k}); "
                f"classify._run_clustering('meteor_efficientnet_b0_model.pth', args)"
            )
            cluster_cmd = [sys.executable, '-c', cluster_code]
            rc = self.run_command(cluster_cmd, cwd=str(k_dir))
            if rc != 0 or self.stop_requested:
                self.msg_queue.put(('status', f"Clustering K={k} failed."))
                continue

            clustered_file = k_dir / 'meteor_efficientnet_b0_model_clustered.pth.zst'
            if not clustered_file.exists():
                self.msg_queue.put(('status', f"Clustered model for K={k} not found."))
                continue

            # The per-K copy of the base .pth is only needed for clustering.
            (k_dir / 'meteor_efficientnet_b0_model.pth').unlink(missing_ok=True)

            # Evaluate on held-out verification data
            self.msg_queue.put(('status', f"Evaluating K={k} on held-out data..."))
            eval_cmd = [
                sys.executable, str(CLASSIFY_PY), '-v', 'evaluate',
                '-m', str(clustered_file),
                str(verify_pos), str(verify_neg),
                '--img-width', '192', '--img-height', '96',
                '--batch-size', str(self.batch_size.get()),
            ]
            eval_stdout, rc = self.run_command_with_output(eval_cmd, cwd=str(k_dir))
            metrics = parse_evaluate_output(eval_stdout) if rc == 0 else {}

            size = clustered_file.stat().st_size
            self.results.append({
                'k': k,
                'path': str(clustered_file),
                'size': size,
                'metrics': metrics,
            })
            self.msg_queue.put(('result', self.results[-1]))

            done_jobs += 1
            self.msg_queue.put(('progress', int(100 * done_jobs / total_jobs)))

        elapsed = timedelta(seconds=int(time.time() - start_time))
        self.msg_queue.put(('status', f"Training run finished in {elapsed}."))
        self.msg_queue.put(('finished', self.results))

    def run_command(self, cmd, cwd):
        """Run a command and stream output to the log. Returns exit code."""
        self.msg_queue.put(('log', f"$ {' '.join(cmd)}"))
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        rc = 1
        try:
            with subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=cwd, env=env, bufsize=1
            ) as proc:
                self.current_proc = proc
                for line in proc.stdout:
                    if self.stop_requested:
                        proc.terminate()
                        break
                    text = line.rstrip('\r\n')
                    if not text:
                        continue
                    is_progress = bool(_PROGRESS_RE.search(text))
                    self.msg_queue.put(('log_overwrite' if is_progress else 'log', text))
                rc = proc.wait()
        except Exception as exc:
            self.msg_queue.put(('log', f"Exception running command: {exc}"))
        finally:
            self.current_proc = None
        return rc

    def run_command_with_output(self, cmd, cwd):
        """Run a command, stream to log, and return full stdout + rc."""
        self.msg_queue.put(('log', f"$ {' '.join(cmd)}"))
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        out = []
        rc = 1
        try:
            with subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=cwd, env=env, bufsize=1
            ) as proc:
                self.current_proc = proc
                for line in proc.stdout:
                    if self.stop_requested:
                        proc.terminate()
                        break
                    text = line.rstrip('\r\n')
                    if not text:
                        continue
                    is_progress = bool(_PROGRESS_RE.search(text))
                    self.msg_queue.put(('log_overwrite' if is_progress else 'log', text))
                    if not is_progress:
                        out.append(line)
                rc = proc.wait()
        except Exception as exc:
            self.msg_queue.put(('log', f"Exception running command: {exc}"))
        finally:
            self.current_proc = None
        return ''.join(out), rc

    # -------------------------------------------------------------------------
    # UI update from worker thread
    # -------------------------------------------------------------------------
    def process_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                kind, payload = msg
                if kind == 'log':
                    self.log(payload)
                elif kind == 'log_overwrite':
                    self.log(payload, overwrite=True)
                elif kind == 'status':
                    self.train_status.configure(text=payload)
                elif kind == 'progress':
                    self.progress['value'] = payload
                elif kind == 'result':
                    self.populate_results_table()
                elif kind == 'finished':
                    self.start_train_btn.configure(state=NORMAL)
                    self.stop_train_btn.configure(state=DISABLED)
                    self.worker_thread = None
                    self.show_results()
                elif kind == 'fetch_status':
                    self.fetch_status.configure(text=payload)
                elif kind == 'fetch_progress':
                    current, total = payload
                    if total > 0:
                        self.fetch_progress.configure(mode='determinate', maximum=total, value=current)
                    else:
                        # Indeterminate: keep pulsing and show count in the status text.
                        pass
                elif kind == 'fetch_done':
                    self.fetch_progress.stop()
                    self.fetch_progress.configure(mode='determinate', value=100)
                    self.fetch_btn.configure(state=NORMAL)
                    self.scan_counts_btn.configure(state=NORMAL)
                    if payload:
                        self.pos_dir.set(payload[0])
                        self.neg_dir.set(payload[1])
                elif kind == 'source_counts':
                    pos_count, neg_count = payload
                    self.source_count_status.configure(
                        text=f"Already fetched: {pos_count} positives, {neg_count} negatives"
                    )
                elif kind == 'prepare_status':
                    self.prepare_status.configure(text=payload)
                elif kind == 'prepare_progress':
                    current, total = payload
                    if total > 0:
                        self.prepare_progress.stop()
                        self.prepare_progress.configure(mode='determinate', maximum=total, value=current)
                elif kind == 'prepare_copy_progress':
                    current, total = payload
                    if total > 0:
                        pct = int(50 + 50 * current / total)
                        self.prepare_progress.stop()
                        self.prepare_progress.configure(mode='determinate', maximum=100, value=pct)
                elif kind == 'prepare_done':
                    self.prepare_progress.stop()
                    self.prepare_progress.configure(mode='determinate', value=100)
                    self.prepare_btn.configure(state=NORMAL)
                    if payload.get('error'):
                        messagebox.showerror("Error", f"Failed to prepare data: {payload['error']}")
                    else:
                        self.set_text(self.data_stats, payload['summary'])
                        self.split_info = payload['split_info']
                    self.update_navigation()
        except queue.Empty:
            pass
        self.after(100, self.process_queue)

    def show_results(self):
        self.populate_results_table()
        self.show_step(4)
        if self.results:
            best = max(self.results, key=lambda r: r['metrics'].get('f1', 0) if 'f1' in r['metrics'] else -1)
            if 'f1' in best['metrics']:
                self.results_summary.configure(
                    text=f"Best model: K={best['k']}, F1={best['metrics']['f1']:.4f}, "
                         f"size={human_size(best['size'])}, path={best['path']}"
                )
            else:
                self.results_summary.configure(text="Training finished, but evaluation metrics were not captured.")

    def populate_results_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        def sort_key(r):
            k = r['k']
            if k == 'current':
                return (0, 0)
            return (1, k)

        for r in sorted(self.results, key=sort_key):
            m = r['metrics']
            k_label = 'Current' if r['k'] == 'current' else r['k']
            self.tree.insert('', 'end', values=(
                k_label,
                human_size(r['size']),
                f"{m.get('f1', 0):.4f}" if 'f1' in m else 'N/A',
                f"{m.get('precision', 0):.4f}" if 'precision' in m else 'N/A',
                f"{m.get('recall', 0):.4f}" if 'recall' in m else 'N/A',
                f"{m.get('threshold', 0):.4f}" if 'threshold' in m else 'N/A',
            ), tags=(r['path'],))

    def on_result_select(self, event):
        sel = self.tree.selection()
        if sel:
            values = self.tree.item(sel[0], 'values')
            self.results_summary.configure(text=f"Selected: K={values[0]}, size={values[1]}, F1={values[2]}")

    def install_model(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Warning", "Please select a model in the results table first.")
            return
        item = self.tree.item(sel[0])
        path = item['tags'][0]
        src = pathlib.Path(path)
        if not src.exists():
            messagebox.showerror("Error", f"Model file not found: {src}")
            return
        dst = MODEL_DIR / 'meteor_efficientnet_b0_model_clustered.pth.zst'
        if src.resolve() == dst.resolve():
            messagebox.showinfo("Info", "The selected model is already the current live model.")
            return
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            backup = dst.with_suffix('.zst.bak')
            counter = 1
            while backup.exists():
                backup = dst.with_suffix(f'.zst.bak{counter}')
                counter += 1
            shutil.copy2(dst, backup)
        shutil.copy2(src, dst)
        messagebox.showinfo("Installed", f"Model installed to:\n{dst}\n\nBackup created if a previous model existed.")

    def save_results_json(self):
        if not self.results:
            messagebox.showwarning("Warning", "No results to save.")
            return
        path = filedialog.asksaveasfilename(
            initialdir=self.output_dir.get(),
            defaultextension='.json',
            filetypes=[('JSON', '*.json')],
            initialfile='retrain_results.json'
        )
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, default=str)
            messagebox.showinfo("Saved", f"Results saved to {path}")


def main():
    app = RetrainApp()
    app.mainloop()


if __name__ == '__main__':
    main()
