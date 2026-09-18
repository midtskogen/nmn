<?php
/**
 * Launcher for the API queue worker daemon.
 *
 * Called by nmn/api/index.php when a CPU-intensive task is enqueued.  It checks
 * whether the worker is already running and, if not, starts it in the
 * background.
 */

require_once __DIR__ . '/api_common.php';
require_once __DIR__ . '/task_queue.php';

function ensure_queue_worker_running() {
    $pid = get_worker_pid();
    if ($pid !== null) {
        return $pid;
    }

    $python = '/usr/bin/python3';
    $script = __DIR__ . '/queue_worker.py';
    $nohup = 'nohup ' . escapeshellarg($python) . ' ' . escapeshellarg($script)
           . ' > ' . escapeshellarg(__DIR__ . '/queue_worker.log')
           . ' 2>&1 &';
    shell_exec($nohup);

    // Give the daemon a moment to write its PID file.
    for ($i = 0; $i < 10; $i++) {
        usleep(100000);
        $pid = get_worker_pid();
        if ($pid !== null) return $pid;
    }
    return null;
}
