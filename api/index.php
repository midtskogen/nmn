<?php
/**
 * Public REST API entry point for NMN data services.
 *
 * Mirrors the functionality of nmn/server/data/index.php but with REST-style
 * paths, API-key authentication for stateful endpoints, IP rate limiting,
 * request logging and a task queue for CPU-intensive work.
 */

require_once __DIR__ . '/api_common.php';
require_once __DIR__ . '/rate_limit.php';
require_once __DIR__ . '/query_logger.php';
require_once __DIR__ . '/task_queue.php';
require_once __DIR__ . '/queue_worker.php';

$start_time = microtime(true);
handle_cors();

$method = $_SERVER['REQUEST_METHOD'];
$path_info = $_SERVER['PATH_INFO'] ?? '';

// Fallback for environments where PATH_INFO is not populated (e.g. some CGI/FastCGI setups).
if ($path_info === '' && isset($_SERVER['REQUEST_URI'])) {
    $script  = $_SERVER['SCRIPT_NAME'] ?? '/api/index.php';
    $uri     = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH) ?: '';
    $prefix  = rtrim(dirname($script), '/');
    if ($prefix !== '' && substr($uri, 0, strlen($prefix)) === $prefix) {
        $path_info = substr($uri, strlen($prefix));
    }
}

if ($path_info === '' && isset($_GET['__path'])) {
    $path_info = '/' . ltrim($_GET['__path'], '/');
}
$path = trim($path_info, '/');
$segments = $path === '' ? [] : explode('/', $path);

// Allow both /api/index.php/v1/stations and /api/v1/stations routing.
if (isset($segments[0]) && $segments[0] === 'v1') {
    array_shift($segments);
}

$client_ip = get_user_ip();
$api_key = get_api_key();
$key_data = null;
$key_id = null;

// ---------------------------------------------------------------------------
// Request context that will be logged at the end.
// ---------------------------------------------------------------------------
$log_entry = [
    'ip'       => $client_ip,
    'key_id'   => null,
    'method'   => $method,
    'endpoint' => $path,
    'station'  => null,
    'status'   => 200,
    'ms'       => 0,
    'bytes'    => 0,
    'quota_hit'=> false,
    'error'    => null,
];

register_shutdown_function(function () use ($start_time, &$log_entry) {
    $log_entry['ms'] = (int) round((microtime(true) - $start_time) * 1000);
    try {
        log_api_request($log_entry);
    } catch (Throwable $e) {
        error_log('API log failed: ' . $e->getMessage());
    }
});

// ---------------------------------------------------------------------------
// Helper: rate-limit with abuse logging.
// ---------------------------------------------------------------------------
function api_check_rate(string $type, string $bucket, ?array $override = null) {
    global $log_entry;
    $result = check_rate_limit($bucket, $type, $override);
    if (!$result['allowed']) {
        $log_entry['status'] = 429;
        log_abuse_event([
            'ip'       => $log_entry['ip'],
            'key_id'   => $log_entry['key_id'],
            'type'     => 'rate_limit',
            'endpoint' => $log_entry['endpoint'],
            'reason'   => $type . ' bucket=' . $bucket,
        ]);
        header('Retry-After: ' . $result['retry_after']);
        api_error('rate_limit_exceeded', 'Too many requests. Please slow down.', 429, ['retry_after' => $result['retry_after']]);
    }
}

// ---------------------------------------------------------------------------
// Helper: require API key for stateful endpoints.
// ---------------------------------------------------------------------------
function api_require_key(string $endpoint_group): array {
    global $key_data, $key_id, $log_entry, $client_ip, $api_key;
    $key_data = validate_api_key($endpoint_group);
    $key_id = $key_data['id'] ?? 'unnamed';
    $log_entry['key_id'] = $key_id;

    $override = $key_data['rate_limit'] ?? null;
    $bucket = $key_id; // rate-limit per key for stateful endpoints
    api_check_rate('stateful', $bucket, $override);
    return $key_data;
}

// ---------------------------------------------------------------------------
// Helper: bounded parameters for queued prediction jobs.
// Anonymous callers may only request modest amounts of work.
// ---------------------------------------------------------------------------
function predict_job_args(): array {
    $args = [];
    if (!empty($_GET['station'])) {
        $stations = array_values(array_filter(array_map('trim', explode(',', $_GET['station']))));
        if (count($stations) > 20) {
            api_error('too_many_stations', 'At most 20 stations may be requested at once.', 400);
        }
        foreach ($stations as $s) validate_station_id($s);
        if ($stations) $args['station'] = implode(',', $stations);
    }
    if (!empty($_GET['days'])) {
        if (!ctype_digit($_GET['days'])) api_error('invalid_days', 'days must be an integer.', 400);
        $days = (int)$_GET['days'];
        if ($days < 1 || $days > 31) api_error('invalid_days', 'days must be between 1 and 31.', 400);
        $args['days'] = $days;
    }
    if (!empty($_GET['start'])) $args['start'] = validate_iso_timestamp($_GET['start']);
    if (!empty($_GET['end']))   $args['end']   = validate_iso_timestamp($_GET['end']);
    if (isset($args['start'], $args['end'])) {
        $span = strtotime($args['end']) - strtotime($args['start']);
        if ($span <= 0 || $span > 45 * 86400) {
            api_error('invalid_range', 'end must be after start and span at most 45 days.', 400);
        }
    }
    return $args;
}

// Pending-queue depth cap: anonymous floods must not grow the queue (or the
// on-disk status files) without bound.
const MAX_PENDING_JOBS = 200;
function predict_enqueue(string $task_id, string $action, array $args, string $bucket) {
    api_check_rate('predict', $bucket);
    if (count(list_pending_jobs()) >= MAX_PENDING_JOBS) {
        log_abuse_event([
            'ip' => $GLOBALS['log_entry']['ip'] ?? 'unknown_ip',
            'type' => 'queue_full', 'endpoint' => $GLOBALS['log_entry']['endpoint'] ?? '',
            'reason' => 'pending queue at ' . MAX_PENDING_JOBS,
        ]);
        api_error('server_busy', 'The job queue is full. Please try again later.', 503);
    }
    write_status_file($task_id, ['status' => 'queued', 'message' => 'status_starting']);
    enqueue_job($task_id, $action, $args);
    ensure_queue_worker_running();
}

// ---------------------------------------------------------------------------
// Task ownership: API-created tasks get an owner sidecar so only the owning
// key (or an admin key) may cancel/stop/control them later.
// ---------------------------------------------------------------------------
function _task_owner_file(string $task_id): string {
    return LOCK_DIR . '/owner_' . $task_id . '.json';
}
function api_record_task_owner(string $task_id, ?string $key_id) {
    global $api_key;
    // Store a hash of the API key itself, not the (optional, non-unique)
    // 'id' field — two keys missing 'id' would otherwise share ownership.
    $owner_hash = hash('sha256', (string)$api_key);
    $ok = file_put_contents(_task_owner_file($task_id),
        json_encode(['owner' => $owner_hash, 'key_id' => $key_id]), LOCK_EX);
    if ($ok === false) {
        // If the sidecar cannot be written the task would be controllable
        // by ANY key in its endpoint group — fail rather than continue.
        api_error('owner_record_failed', 'Could not record task ownership.', 500);
    }
}
function api_require_task_ownership(string $task_id) {
    global $api_key;
    $f = _task_owner_file($task_id);
    if (!file_exists($f)) return; // not created through the API
    $rec = json_decode((string) file_get_contents($f), true) ?: [];
    $owner = $rec['owner'] ?? null;
    if ($owner !== null && hash_equals($owner, hash('sha256', (string)$api_key))) return;
    validate_api_key('admin');
}

// Unguessable task ids (uniqid() is microtime-derived and enumerable).
function new_task_id(string $prefix): string {
    return $prefix . '_' . bin2hex(random_bytes(8));
}

// ---------------------------------------------------------------------------
// Shared download-job launcher: payload JSON -> temp file -> coordinator.
// ---------------------------------------------------------------------------
function api_start_download(string $raw, string $client_ip, ?string $key_id, bool $wait = false) {
    global $log_entry;
    $max_payload = 5 * 1024 * 1024;
    if (strlen($raw) > $max_payload) {
        $log_entry['status'] = 413;
        api_error('payload_too_large', 'Download payload too large.', 413);
    }
    // Reuse the web UI concurrent-download semaphore.
    $max_concurrent = 8;
    $sem_file = LOCK_DIR . '/download_semaphore.lock';
    $sem = fopen($sem_file, 'c');
    if (!$sem || !flock($sem, LOCK_EX)) {
        api_error('server_busy', 'Could not acquire download lock.', 503);
    }
    $lock_files = glob(LOCK_DIR . '/master_task_*.lock');
    if (count($lock_files) >= $max_concurrent) {
        flock($sem, LOCK_UN); fclose($sem);
        api_error('server_busy', 'Too many concurrent downloads.', 503);
    }
    $task_id = new_task_id('master_task');
    api_record_task_owner($task_id, $key_id);
    $payload_file = tempnam(LOCK_DIR, 'payload_');
    file_put_contents($payload_file, $raw, LOCK_EX);
    touch(LOCK_DIR . '/' . $task_id . '.lock');
    flock($sem, LOCK_UN); fclose($sem);

    $cmd = python_exec() . ' ' . escapeshellarg(DATA_DIR . '/controller.py') . ' download '
         . escapeshellarg($task_id) . ' ' . escapeshellarg($payload_file) . ' ' . escapeshellarg($client_ip)
         . ' > /dev/null 2>&1 &';
    shell_exec($cmd);

    // Synchronous mode (GET /download): hold the request until the
    // coordinator finishes so callers get the final file list directly.
    // If the wait exceeds the budget, fall back to the async handle.
    if ($wait) {
        // Apache TimeOut / FcgidIOTimeout is 300s — leave margin to respond.
        set_time_limit(295);
        $deadline = microtime(true) + 280;
        while (microtime(true) < $deadline) {
            usleep(500000);
            $data = read_status_file($task_id);
            if (is_array($data) && in_array($data['status'] ?? '', ['complete', 'error'], true)) {
                $data['task_id'] = $task_id;
                api_json_response($data, ($data['status'] === 'error') ? 500 : 200);
            }
        }
        api_json_response(['task_id' => $task_id, 'status' => 'pending', 'poll_url' => '/api/v1/downloads/' . $task_id], 202);
    }
    api_json_response(['task_id' => $task_id, 'status' => 'pending', 'poll_url' => '/api/v1/downloads/' . $task_id], 202);
}

// ---------------------------------------------------------------------------
// Helper: run a Python script synchronously and return JSON.
// ---------------------------------------------------------------------------
function run_python(array $args) {
    global $log_entry;
    $cmd = array_merge([python_exec()], array_map('escapeshellarg', $args));
    $output = shell_exec(implode(' ', $cmd));
    if ($output === null) {
        $log_entry['status'] = 500;
        api_error('backend_error', 'Backend produced no output.', 500);
    }
    // Decode without forcing associative arrays so empty Python dicts stay JSON
    // objects instead of being turned into empty JSON arrays.
    $data = json_decode($output);
    if (json_last_error() !== JSON_ERROR_NONE) {
        // Some endpoints return raw JSON strings, but decode should still work.
        $log_entry['status'] = 500;
        api_error('invalid_backend_response', 'Backend returned invalid JSON.', 500);
    }
    return $data;
}

// ---------------------------------------------------------------------------
// Helper: generic task status read from the shared locks dir.
// ---------------------------------------------------------------------------
function read_status_file(string $task_id): ?array {
    $file = status_file_path($task_id);
    if (!file_exists($file)) return null;
    $raw = file_get_contents($file);
    $data = json_decode($raw, true);
    return is_array($data) ? $data : null;
}

// ---------------------------------------------------------------------------
// Read-only rate limit applies before any anonymous endpoint.
// ---------------------------------------------------------------------------
if (empty($segments)) {
    header('Location: /api/v1/docs', true, 302);
    exit;
}

$resource = $segments[0] ?? '';

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------
switch ($resource) {

    // --- Read-only endpoints ------------------------------------------------
    case 'stations':
        if (count($segments) >= 3 && $segments[1] !== '' && $segments[2] === 'stats') {
            api_check_rate('read_only', $client_ip);
            $station_id = validate_station_id($segments[1]);
            $log_entry['station'] = $station_id;
            $start_date = isset($_GET['start_date']) ? validate_date($_GET['start_date']) : '';
            $end_date   = isset($_GET['end_date'])   ? validate_date($_GET['end_date'])   : '';
            $args = [DATA_DIR . '/controller.py', 'get_station_stats', $station_id];
            if ($start_date !== '') $args[] = $start_date;
            if ($start_date !== '' && $end_date !== '') $args[] = $end_date;
            $data = run_python($args);
            api_json_response($data);
        }
        api_check_rate('read_only', $client_ip);
        $data = run_python([DATA_DIR . '/controller.py', 'get_stations']);
        api_json_response($data);

    case 'cameras':
        if (($segments[1] ?? '') !== 'fovs') {
            api_error('not_found', 'Unknown endpoint.', 404);
        }
        api_check_rate('read_only', $client_ip);
        $data = run_python([DATA_DIR . '/controller.py', 'get_camera_fovs']);
        api_json_response($data);

    case 'kp':
        api_check_rate('read_only', $client_ip);
        $data = run_python([DATA_DIR . '/controller.py', 'get_kp_data']);
        api_json_response($data);

    case 'lightning':
        api_check_rate('read_only', $client_ip);
        $date = isset($_GET['date']) ? validate_date($_GET['date']) : date('Y-m-d');
        $data = run_python([DATA_DIR . '/controller.py', 'get_lightning_data', $date]);
        api_json_response($data);

    case 'meteors':
        api_check_rate('read_only', $client_ip);
        $data = run_python([DATA_DIR . '/controller.py', 'get_meteor_data']);
        api_json_response($data);

    // --- Predictions (queued) -----------------------------------------------
    case 'predict':
        $sub = $segments[1] ?? '';
        if ($sub === 'passes') {
            if ($method === 'POST' && count($segments) === 2) {
                $task_id = new_task_id('pass_task');
                predict_enqueue($task_id, 'find_passes', predict_job_args(), $client_ip);
                api_json_response(['task_id' => $task_id, 'status' => 'queued', 'poll_url' => '/api/v1/predict/passes/' . $task_id], 202);
            }
            if ($method === 'GET' && count($segments) === 3) {
                api_check_rate('read_only', $client_ip);
                $task_id = validate_task_id($segments[2], ['pass_task']);
                $data = read_status_file($task_id);
                if ($data === null) api_error('task_not_found', 'Task not found or not started yet.', 404);
                api_json_response($data);
            }
        }
        if ($sub === 'aircraft') {
            if ($method === 'POST' && count($segments) === 2) {
                $task_id = new_task_id('aircraft_task');
                predict_enqueue($task_id, 'find_aircraft_crossings', predict_job_args(), $client_ip);
                api_json_response(['task_id' => $task_id, 'status' => 'queued', 'poll_url' => '/api/v1/predict/aircraft/' . $task_id], 202);
            }
            if ($method === 'GET' && count($segments) === 3) {
                api_check_rate('read_only', $client_ip);
                $task_id = validate_task_id($segments[2], ['aircraft_task']);
                $data = read_status_file($task_id);
                if ($data === null) api_error('task_not_found', 'Task not found or not started yet.', 404);
                api_json_response($data);
            }
        }
        api_error('not_found', 'Unknown endpoint.', 404);

    // --- Downloads ------------------------------------------------------------
    case 'downloads':
        if ($method === 'POST' && count($segments) === 1) {
            api_require_key('download');
            $raw = file_get_contents('php://input');
            $json = json_decode($raw, true);
            if (!is_array($json)) {
                api_error('invalid_json', 'Request body must be JSON.', 400);
            }
            api_start_download($raw, $client_ip, $key_id);
        }
        if ($method === 'GET' && count($segments) === 2) {
            api_check_rate('read_only', $client_ip);
            $task_id = validate_task_id($segments[1], ['master_task']);
            $data = read_status_file($task_id);
            if ($data === null) api_json_response(['status' => 'pending']);
            api_json_response($data);
        }
        if ($method === 'DELETE' && count($segments) === 2) {
            api_require_key('download');
            $task_id = validate_task_id($segments[1], ['master_task']);
            api_require_task_ownership($task_id);
            $cmd = python_exec() . ' ' . escapeshellarg(DATA_DIR . '/controller.py') . ' cancel ' . escapeshellarg($task_id);
            shell_exec($cmd);
            api_json_response(['success' => true, 'task_id' => $task_id, 'status' => 'cancelled']);
        }
        api_error('not_found', 'Unknown endpoint.', 404);

    // --- Simple download: one URL, no JSON body, synchronous -----------------
    // GET /api/v1/download?station=ams172&camera=2&date=YYYY-MM-DD&hour=H&minute=M
    //     [&file_type=lowres] [&length=N] [&interval=N] [&duration=N]
    //     [&stitch_equirect=1] [&stitch_fisheye=1] [&lang=en]
    // Waits for the download to finish and returns the final status (files +
    // errors).  Jobs still running after 5 min return the async 202 handle.
    // File types: lowres|hires (video), image|image_lowres (still),
    //             image_long|image_lowres_long (stacked still),
    //             timelapse|timelapse_hires (full-day stitched videos;
    //             hour/minute unused, length=days, interval=day step).
    case 'download':
        if ($method !== 'GET' || count($segments) !== 1) api_error('not_found', 'Unknown endpoint.', 404);
        api_require_key('download');

        $stations = array_values(array_filter(array_map('trim', explode(',', (string)($_GET['station'] ?? '')))));
        if (!$stations) api_error('missing_parameter', 'station is required (comma-separated amsNNN ids).', 400);
        if (count($stations) > 10) api_error('too_many_stations', 'At most 10 stations per request.', 400);
        foreach ($stations as $s) validate_station_id($s);

        $file_types = ['lowres', 'hires', 'image', 'image_lowres', 'image_long',
                       'image_lowres_long', 'timelapse', 'timelapse_hires'];
        $file_type = (string)($_GET['file_type'] ?? 'lowres');
        if (!in_array($file_type, $file_types, true)) {
            api_error('invalid_file_type', 'file_type must be one of: ' . implode(', ', $file_types), 400);
        }
        $is_timelapse = str_starts_with($file_type, 'timelapse');

        // Timelapse downloads pick cameras via stitch_equirect/fisheye, so
        // camera is only required for the other file types.
        $cameras = array_values(array_filter(array_map('trim', explode(',', (string)($_GET['camera'] ?? ''))), 'strlen'));
        if (!$cameras && !$is_timelapse) api_error('missing_parameter', 'camera is required (comma-separated numbers).', 400);
        $cameras = array_map(fn($c) => validate_camera_num($c), $cameras);

        $date = isset($_GET['date']) ? validate_date($_GET['date'])
            : api_error('missing_parameter', 'date is required (YYYY-MM-DD).', 400);

        $hour = $minute = 0;
        if (!$is_timelapse) {
            if (!isset($_GET['hour'], $_GET['minute']) || !ctype_digit((string)$_GET['hour']) || !ctype_digit((string)$_GET['minute'])
                || (int)$_GET['hour'] > 23 || (int)$_GET['minute'] > 59) {
                api_error('invalid_parameter', 'hour (0-23) and minute (0-59) are required.', 400);
            }
            $hour = (int)$_GET['hour']; $minute = (int)$_GET['minute'];
        }

        // The coordinator enforces its own bounds; mirror the same limits here
        // so bad URLs fail fast with a clear 400.
        $bounds = ['length' => 100, 'interval' => 365, 'duration' => 60];
        foreach ($bounds as $name => $max) {
            $v = $_GET[$name] ?? '1';
            if (!ctype_digit((string)$v) || (int)$v < 1 || (int)$v > $max) {
                api_error('invalid_parameter', "$name must be an integer 1-$max.", 400);
            }
        }

        $payload = [
            'stations'   => $stations,
            'cameras'    => $cameras,
            'file_type'  => $file_type,
            'date'       => $date,
            'hour'       => str_pad((string)$hour, 2, '0', STR_PAD_LEFT),
            'minute'     => str_pad((string)$minute, 2, '0', STR_PAD_LEFT),
            'length'     => (int)($_GET['length'] ?? 1),
            'interval'   => (int)($_GET['interval'] ?? 1),
            'duration'   => (int)($_GET['duration'] ?? 1),
        ];
        foreach (['stitch_equirect', 'stitch_fisheye'] as $flag) {
            if (!empty($_GET[$flag])) $payload[$flag] = in_array($_GET[$flag], ['1', 'true', 'yes'], true);
        }
        if (isset($_GET['lang']) && preg_match('/^[a-z]{2}(_[A-Z]{2})?$/', $_GET['lang'])) {
            $payload['lang'] = $_GET['lang'];
        }
        api_start_download(json_encode($payload, JSON_THROW_ON_ERROR), $client_ip, $key_id, true);

    // --- Streams --------------------------------------------------------------
    case 'streams':
        if ($method === 'POST' && count($segments) === 1) {
            api_require_key('streams');
            $station_id = isset($_POST['station_id']) ? validate_station_id($_POST['station_id']) : (isset($_GET['station_id']) ? validate_station_id($_GET['station_id']) : '');
            $camera_num = isset($_POST['camera_num']) ? validate_camera_num($_POST['camera_num']) : (isset($_GET['camera_num']) ? validate_camera_num($_GET['camera_num']) : 0);
            $resolution = isset($_POST['resolution']) ? $_POST['resolution'] : ($_GET['resolution'] ?? 'lowres');
            if (!in_array($resolution, ['lowres', 'hires'], true)) $resolution = 'lowres';
            $hevc = isset($_POST['hevc_supported']) ? $_POST['hevc_supported'] : ($_GET['hevc_supported'] ?? 'false');

            // Bound concurrent streams: each spawns an ssh tunnel plus an
            // ffmpeg relay on this server and a camera on the station.
            $active_streams = 0;
            foreach (glob(LOCK_DIR . '/stream_*.json') ?: [] as $sf) {
                if (time() - filemtime($sf) < 15 * 60) $active_streams++;
            }
            if ($active_streams >= 8) {
                api_error('server_busy', 'Too many concurrent streams. Please try again later.', 503);
            }

            $task_id = new_task_id('stream');
            api_record_task_owner($task_id, $key_id);
            $cmd = python_exec() . ' ' . escapeshellarg(DATA_DIR . '/controller.py') . ' _internal_start_stream '
                 . escapeshellarg($task_id) . ' '
                 . escapeshellarg($station_id) . ' '
                 . escapeshellarg($camera_num) . ' '
                 . escapeshellarg($resolution) . ' '
                 . escapeshellarg($hevc) . ' '
                 . escapeshellarg($client_ip) . ' > /dev/null 2>&1 &';
            shell_exec($cmd);
            api_json_response(['task_id' => $task_id, 'status' => 'pending', 'poll_url' => '/api/v1/streams/' . $task_id], 202);
        }
        if ($method === 'GET' && count($segments) === 2) {
            api_check_rate('read_only', $client_ip);
            $task_id = validate_task_id($segments[1], ['stream']);
            $data = read_status_file($task_id);
            if ($data === null) api_json_response(['status' => 'pending']);
            api_json_response($data);
        }
        if ($method === 'DELETE' && count($segments) === 2) {
            api_require_key('streams');
            $task_id = validate_task_id($segments[1], ['stream']);
            api_require_task_ownership($task_id);
            $cmd = python_exec() . ' ' . escapeshellarg(DATA_DIR . '/controller.py') . ' stop_stream ' . escapeshellarg($task_id);
            shell_exec($cmd);
            api_json_response(['success' => true, 'task_id' => $task_id, 'status' => 'stop_requested']);
        }
        if ($method === 'POST' && count($segments) === 3 && $segments[2] === 'transcode') {
            api_require_key('streams');
            $task_id = validate_task_id($segments[1], ['stream']);
            api_require_task_ownership($task_id);
            $data = run_python([DATA_DIR . '/controller.py', 'request_transcode', $task_id]);
            api_json_response($data);
        }
        api_error('not_found', 'Unknown endpoint.', 404);

    // --- Overlays / archive / enhance (require key) -------------------------
    case 'grids':
        api_require_key('grids');
        if (count($segments) !== 3) api_error('not_found', 'Unknown endpoint.', 404);
        $station_id = validate_station_id($segments[1]);
        $camera_num = validate_camera_num($segments[2]);
        $data = run_python([DATA_DIR . '/controller.py', 'fetch_grid', '', $station_id, $camera_num, $client_ip]);
        api_json_response($data);

    case 'annotations':
        api_require_key('annotations');
        if (count($segments) !== 3) api_error('not_found', 'Unknown endpoint.', 404);
        $station_id = validate_station_id($segments[1]);
        $camera_num = validate_camera_num($segments[2]);
        $data = run_python([DATA_DIR . '/controller.py', 'fetch_annotation', '', $station_id, $camera_num, $client_ip]);
        api_json_response($data);

    case 'archive':
        api_require_key('archive');
        $sub = $segments[1] ?? '';
        if ($sub === 'grid') {
            $station_id = isset($_GET['station_id']) ? validate_station_id($_GET['station_id']) : api_error('missing_parameter', 'station_id required', 400);
            $camera_num = isset($_GET['camera_num']) ? validate_camera_num($_GET['camera_num']) : api_error('missing_parameter', 'camera_num required', 400);
            $timestamp = isset($_GET['timestamp']) ? validate_iso_timestamp($_GET['timestamp']) : api_error('missing_parameter', 'timestamp required', 400);
            $data = run_python([DATA_DIR . '/controller.py', 'fetch_archive_grid', $station_id, $camera_num, $timestamp, $client_ip]);
            api_json_response($data);
        }
        if ($sub === 'annotation') {
            $station_id = isset($_GET['station_id']) ? validate_station_id($_GET['station_id']) : api_error('missing_parameter', 'station_id required', 400);
            $camera_num = isset($_GET['camera_num']) ? validate_camera_num($_GET['camera_num']) : api_error('missing_parameter', 'camera_num required', 400);
            $timestamp = isset($_GET['timestamp']) ? validate_iso_timestamp($_GET['timestamp']) : api_error('missing_parameter', 'timestamp required', 400);
            $data = run_python([DATA_DIR . '/controller.py', 'fetch_archive_annotation', $station_id, $camera_num, $timestamp, $client_ip]);
            api_json_response($data);
        }
        if ($sub === 'mask') {
            $station_id = isset($_GET['station_id']) ? validate_station_id($_GET['station_id']) : api_error('missing_parameter', 'station_id required', 400);
            $camera_num = isset($_GET['camera_num']) ? validate_camera_num($_GET['camera_num']) : api_error('missing_parameter', 'camera_num required', 400);
            $timestamp = isset($_GET['timestamp']) ? validate_iso_timestamp($_GET['timestamp']) : api_error('missing_parameter', 'timestamp required', 400);
            $data = run_python([DATA_DIR . '/controller.py', 'fetch_archive_mask', $station_id, $camera_num, $timestamp, $client_ip]);
            api_json_response($data);
        }
        api_error('not_found', 'Unknown endpoint.', 404);

    case 'enhance':
        if ($method !== 'POST') api_error('method_not_allowed', 'POST required.', 405);
        api_require_key('enhance');
        $max_payload = 256 * 1024;
        $raw = file_get_contents('php://input');
        if (strlen($raw) > $max_payload) {
            $log_entry['status'] = 413;
            api_error('payload_too_large', 'Enhance payload too large.', 413);
        }
        $payload = json_decode($raw, true);
        if (!is_array($payload) || empty($payload['image']) || !isset($payload['filter'])
            || !is_string($payload['image'])) {
            api_error('invalid_json', 'JSON body must contain image and filter.', 400);
        }
        if (!preg_match('/^[A-Za-z0-9_.\/-]+$/', $payload['image'])
            || strpos($payload['image'], '..') !== false) {
            api_error('invalid_image', 'Invalid image name.', 400);
        }
        // Bound the number of queued enhance jobs to avoid queue flooding.
        $pending = 0;
        foreach (glob(LOCK_DIR . '/queue_*.json') ?: [] as $qf) {
            if (time() - filemtime($qf) < 3600) $pending++;
        }
        if ($pending >= 100) {
            api_error('server_busy', 'Too many pending tasks. Please try again later.', 503);
        }
        $task_id = new_task_id('api_task');
        api_record_task_owner($task_id, $key_id);
        $args = [
            'image'  => basename($payload['image']),
            'filter' => (int)$payload['filter'],
        ];
        write_status_file($task_id, ['status' => 'queued', 'message' => 'status_starting']);
        enqueue_job($task_id, 'enhance_filter', $args);
        ensure_queue_worker_running();
        api_json_response(['task_id' => $task_id, 'status' => 'queued', 'poll_url' => '/api/v1/tasks/' . $task_id], 202);

    // --- Generic task status --------------------------------------------------
    case 'tasks':
        if ($method !== 'GET' || count($segments) !== 2) api_error('not_found', 'Unknown endpoint.', 404);
        api_check_rate('read_only', $client_ip);
        $task_id = validate_task_id($segments[1], ['api_task']);
        $data = read_status_file($task_id);
        if ($data === null) api_json_response(['status' => 'pending']);
        api_json_response($data);

    // --- Admin / stats --------------------------------------------------------
    case 'admin':
        if (($segments[1] ?? '') === 'stats') {
            // Only authenticated keys can view stats.  No anonymous access.
            api_require_key('admin');
            require_once __DIR__ . '/stats.php';
            exit;
        }
        api_error('not_found', 'Unknown endpoint.', 404);

    // --- Docs -----------------------------------------------------------------
    case 'docs':
        $sub = $segments[1] ?? '';
        if ($sub === 'openapi.yaml') {
            $file = __DIR__ . '/openapi.yaml';
        } elseif ($sub === 'index.html' || $sub === 'swagger') {
            $file = __DIR__ . '/docs/index.html';
        } else {
            $file = __DIR__ . '/docs/README.html';
        }
        if (!file_exists($file)) api_error('not_found', 'Documentation not available.', 404);
        if (substr($file, -5) === '.yaml') {
            header('Content-Type: application/yaml');
        } else {
            header('Content-Type: text/html');
        }
        readfile($file);
        exit;

    default:
        api_error('not_found', 'Unknown endpoint.', 404);
}
