<?php
/**
 * Simple on-disk task queue for CPU-intensive API jobs.
 *
 * Jobs are stored as NDJSON in nmn/api/task_queue.jsonl.  Workers claim pending
 * jobs, update their status and remove finished jobs during cleanup.
 */

if (!defined('TASK_QUEUE_FILE')) {
    define('TASK_QUEUE_FILE', __DIR__ . '/task_queue.jsonl');
}
if (!defined('QUEUE_WORKER_PID_FILE')) {
    define('QUEUE_WORKER_PID_FILE', __DIR__ . '/queue_worker.pid');
}

function _queue_dir() {
    $dir = dirname(TASK_QUEUE_FILE);
    if (!is_dir($dir)) { mkdir($dir, 0775, true); }
    return $dir;
}

function _read_queue(): array {
    _queue_dir();
    if (!file_exists(TASK_QUEUE_FILE)) return [];
    $jobs = [];
    $handle = fopen(TASK_QUEUE_FILE, 'r');
    if (!$handle) return [];
    while (($line = fgets($handle)) !== false) {
        $line = trim($line);
        if ($line === '') continue;
        $job = json_decode($line, true);
        if (is_array($job)) $jobs[] = $job;
    }
    fclose($handle);
    return $jobs;
}

function _write_queue(array $jobs) {
    _queue_dir();
    $lines = [];
    foreach ($jobs as $job) {
        $lines[] = json_encode($job, JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR);
    }
    $tmp = TASK_QUEUE_FILE . '.tmp.' . uniqid('', true);
    file_put_contents($tmp, implode("\n", $lines) . (count($lines) ? "\n" : ''), LOCK_EX);
    rename($tmp, TASK_QUEUE_FILE);
}

/**
 * Add a job to the queue.
 */
function enqueue_job(string $task_id, string $action, array $args = [], array $extra = []): array {
    $job = array_merge([
        'task_id'   => $task_id,
        'action'    => $action,
        'args'      => $args,
        'status'    => 'pending',
        'created_ts'=> time(),
        'started_ts'=> null,
        'finished_ts'=> null,
        'pid'       => null,
        'exit_code' => null,
        'error'     => null,
    ], $extra);

    $fp = fopen(TASK_QUEUE_FILE, 'a');
    if (!$fp || !flock($fp, LOCK_EX)) {
        throw new Exception('Could not acquire task queue lock');
    }
    fwrite($fp, json_encode($job, JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR) . "\n");
    fflush($fp);
    flock($fp, LOCK_UN);
    fclose($fp);
    return $job;
}

/**
 * Find a job by task_id.
 */
function get_job(string $task_id): ?array {
    foreach (_read_queue() as $job) {
        if ($job['task_id'] === $task_id) return $job;
    }
    return null;
}

/**
 * Update a job's fields atomically.
 */
function update_job(string $task_id, array $updates) {
    $jobs = _read_queue();
    $found = false;
    foreach ($jobs as &$job) {
        if ($job['task_id'] === $task_id) {
            $job = array_merge($job, $updates);
            $found = true;
            break;
        }
    }
    if ($found) {
        _write_queue($jobs);
    }
}

/**
 * Return pending jobs in FIFO order.
 */
function list_pending_jobs(): array {
    return array_values(array_filter(_read_queue(), fn($j) => $j['status'] === 'pending'));
}

/**
 * Return currently running jobs.
 */
function list_running_jobs(): array {
    return array_values(array_filter(_read_queue(), fn($j) => $j['status'] === 'running'));
}

/**
 * Return summary counts by status.
 */
function queue_summary(): array {
    $counts = ['pending' => 0, 'running' => 0, 'done' => 0, 'error' => 0];
    foreach (_read_queue() as $job) {
        $counts[$job['status']] = ($counts[$job['status']] ?? 0) + 1;
    }
    return $counts;
}

/**
 * Remove finished jobs older than $age_seconds.
 */
function cleanup_finished_jobs(int $age_seconds = 86400) {
    $now = time();
    $jobs = array_values(array_filter(_read_queue(), function ($job) use ($now, $age_seconds) {
        if (in_array($job['status'], ['done', 'error'], true)) {
            $finished = $job['finished_ts'] ?? $job['started_ts'] ?? $job['created_ts'];
            return ($now - $finished) < $age_seconds;
        }
        return true;
    }));
    _write_queue($jobs);
}

/**
 * Reset stale running jobs to pending.
 */
function reset_stale_jobs(int $timeout_seconds = 1800) {
    $now = time();
    $jobs = _read_queue();
    $changed = false;
    foreach ($jobs as &$job) {
        if ($job['status'] === 'running') {
            $started = $job['started_ts'] ?? $job['created_ts'];
            $pid = $job['pid'] ?? null;
            $pid_alive = $pid ? posix_kill((int)$pid, 0) : false;
            if (!$pid_alive && ($now - $started) > 60) {
                $job['status'] = 'pending';
                $job['started_ts'] = null;
                $job['pid'] = null;
                $job['error'] = 'worker_died';
                $changed = true;
            }
        }
    }
    if ($changed) _write_queue($jobs);
}

/**
 * Write the queue worker PID file.
 */
function set_worker_pid(int $pid) {
    file_put_contents(QUEUE_WORKER_PID_FILE, $pid, LOCK_EX);
}

/**
 * Read the queue worker PID if it is alive.
 */
function get_worker_pid(): ?int {
    if (!file_exists(QUEUE_WORKER_PID_FILE)) return null;
    $pid = (int) trim(file_get_contents(QUEUE_WORKER_PID_FILE));
    if ($pid > 0 && posix_kill($pid, 0)) return $pid;
    return null;
}
