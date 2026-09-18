<?php
/**
 * Common helpers for the NMN public API.
 *
 * This file is not a web entry point; it is included by index.php and stats.php.
 * It handles environment setup, config/key loading, IP extraction, CORS, JSON
 * responses, input validation and basic quota/rate-limit helpers.
 */

// --- Environment / paths ---
// The API entry point lives in nmn/api/.  The live runtime data directory is
// the project-root data/ directory (where cameras.json, stations.json,
// access_log.json, quota_tracker.json, etc. live).  The Python scripts live in
// nmn/server/data and are invoked with NMN_DATA_DIR set to the runtime dir.
function api_resolve_data_dir(): string {
    // Allow explicit override.
    if (!empty($_SERVER['NMN_DATA_DIR']) && is_dir($_SERVER['NMN_DATA_DIR'])) {
        return rtrim($_SERVER['NMN_DATA_DIR'], '/');
    }
    // If DOCUMENT_ROOT is set, try <docroot>/data first (mirrors the web UI).
    if (!empty($_SERVER['DOCUMENT_ROOT'])) {
        $candidate = rtrim($_SERVER['DOCUMENT_ROOT'], '/') . '/data';
        if (is_dir($candidate)) return $candidate;
    }
    // Project-root data: nmn/api -> nmn -> project-root -> data
    $project_root = dirname(dirname(__DIR__));
    $candidate = $project_root . '/data';
    if (is_dir($candidate)) return $candidate;
    // Final fallback to the source tree's data directory.
    $candidate = realpath(__DIR__ . '/../server/data') ?: __DIR__ . '/../server/data';
    return $candidate;
}

if (!defined('API_ROOT')) {
    define('API_ROOT', __DIR__);
}
if (!defined('DATA_DIR')) {
    define('DATA_DIR', api_resolve_data_dir());
}
if (!defined('SECRETS_DIR')) {
    // Prefer an explicit env-configured directory OUTSIDE the web root.
    // The legacy default resolves to <webroot>/etc on a docroot deployment,
    // which lets Apache serve api_keys.json/credentials.json to anyone.
    $secrets = getenv('NMN_SECRETS_DIR') ?: '';
    if ($secrets === '' || !is_dir($secrets)) {
        $secrets = realpath(__DIR__ . '/../../etc') ?: __DIR__ . '/../../etc';
    }
    define('SECRETS_DIR', rtrim($secrets, '/'));
}
if (!defined('LOCK_DIR')) {
    define('LOCK_DIR', DATA_DIR . '/locks');
}

putenv('NMN_DATA_DIR=' . DATA_DIR);
putenv('NMN_LOCK_DIR=' . LOCK_DIR);
if (is_dir(SECRETS_DIR)) {
    putenv('NMN_CONFIG_FILE=' . SECRETS_DIR . '/config.json');
    putenv('NMN_CREDENTIALS_FILE=' . SECRETS_DIR . '/credentials.json');
}

$PYTHON_EXECUTABLE = '/usr/bin/python3';
$PYTHON_SCRIPT = DATA_DIR . '/controller.py';
$SATELLITE_SCRIPT = DATA_DIR . '/predict_sat.py';
$AIRCRAFT_SCRIPT = DATA_DIR . '/predict_flight.py';

if (!is_dir(LOCK_DIR)) { mkdir(LOCK_DIR, 0775, true); }

// --- Config & keys ---
$_API_CONFIG = null;
$_API_KEYS = null;

function load_api_config() {
    global $_API_CONFIG;
    if ($_API_CONFIG !== null) return $_API_CONFIG;
    $path = SECRETS_DIR . '/api_config.json';
    if (!is_readable($path)) return [];
    $data = json_decode(file_get_contents($path), true);
    $_API_CONFIG = is_array($data) ? $data : [];
    return $_API_CONFIG;
}

function load_api_keys() {
    global $_API_KEYS;
    if ($_API_KEYS !== null) return $_API_KEYS;
    $path = SECRETS_DIR . '/api_keys.json';
    if (!is_readable($path)) return [];
    $data = json_decode(file_get_contents($path), true);
    if (!is_array($data) || !isset($data['keys']) || !is_array($data['keys'])) {
        $_API_KEYS = [];
    } else {
        $_API_KEYS = $data['keys'];
    }
    return $_API_KEYS;
}

// --- IP extraction ---
function get_user_ip() {
    // Extra trusted proxy IPs can be configured via NMN_TRUSTED_PROXIES
    // (comma-separated, e.g. "172.22.0.1" for a docker reverse-proxy).
    // Without it every request appears to come from the proxy address, so
    // per-IP rate limits degenerate to one global bucket and abuse cannot
    // be attributed to a real client.
    $trusted_proxies = ['127.0.0.1', '::1'];
    foreach (explode(',', (string) getenv('NMN_TRUSTED_PROXIES')) as $p) {
        $p = trim($p);
        if ($p !== '' && filter_var($p, FILTER_VALIDATE_IP)) $trusted_proxies[] = $p;
    }
    $remote = $_SERVER['REMOTE_ADDR'] ?? '';
    if (in_array($remote, $trusted_proxies, true)) {
        if (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {
            $candidate = trim(explode(',', $_SERVER['HTTP_X_FORWARDED_FOR'])[0]);
            if (filter_var($candidate, FILTER_VALIDATE_IP)) return $candidate;
        }
        if (!empty($_SERVER['HTTP_X_REAL_IP'])) {
            $candidate = trim($_SERVER['HTTP_X_REAL_IP']);
            if (filter_var($candidate, FILTER_VALIDATE_IP)) return $candidate;
        }
    }
    return $remote !== '' ? $remote : 'unknown_ip';
}

// --- CORS ---
function handle_cors() {
    $cfg = load_api_config();
    $allowed = $cfg['allowed_origins'] ?? [];
    $allow_wildcard = !empty($cfg['allow_wildcard_origin']);
    $origin = $_SERVER['HTTP_ORIGIN'] ?? '';

    $send_headers = false;
    if ($allow_wildcard && $origin !== '') {
        header('Access-Control-Allow-Origin: *');
        $send_headers = true;
    } elseif ($origin !== '' && in_array($origin, $allowed, true)) {
        header('Access-Control-Allow-Origin: ' . $origin);
        header('Vary: Origin');
        $send_headers = true;
    }

    if ($send_headers) {
        header('Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS');
        header('Access-Control-Allow-Headers: Content-Type, X-API-Key, X-CSRF-Token');
        header('Access-Control-Max-Age: 86400');
    }

    if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
        http_response_code(204);
        exit;
    }
}

// --- JSON responses ---
function api_json_response($data, $status = 200) {
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    $out = json_encode($data, JSON_THROW_ON_ERROR | JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE) . "\n";
    // Record the real outcome in the request log (previously every error
    // was logged as the 200 default, hiding rejected/abusive requests).
    if (isset($GLOBALS['log_entry']) && is_array($GLOBALS['log_entry'])) {
        $GLOBALS['log_entry']['status'] = $status;
        $GLOBALS['log_entry']['bytes'] = strlen($out);
    }
    echo $out;
    exit;
}

function api_error($code, $message, $status = 400, $extra = []) {
    api_json_response(array_merge(['error' => $code, 'message' => $message], $extra), $status);
}

// --- API key validation ---
function get_api_key() {
    // Keys are accepted via the X-API-Key header or the POST body only.
    // A ?api_key= query parameter would end up in Apache access logs,
    // reverse-proxy logs, browser history and Referer headers.
    $header = $_SERVER['HTTP_X_API_KEY'] ?? '';
    if ($header !== '') return $header;
    return $_POST['api_key'] ?? '';
}

/**
 * Report a failed authentication: log it to the abuse log and throttle the
 * caller so API keys cannot be brute-forced at full request rate.
 */
function _api_auth_failed(string $code, string $message, int $status) {
    if (function_exists('check_rate_limit') && function_exists('log_abuse_event')) {
        $ip = get_user_ip();
        log_abuse_event([
            'ip'       => $ip,
            'key_id'   => null,
            'type'     => 'auth_fail',
            'endpoint' => $GLOBALS['log_entry']['endpoint'] ?? '',
            'reason'   => $code,
        ]);
        $result = check_rate_limit($ip, 'auth_fail');
        if (!$result['allowed']) {
            header('Retry-After: ' . $result['retry_after']);
            api_error('rate_limit_exceeded', 'Too many failed authentication attempts.', 429,
                      ['retry_after' => $result['retry_after']]);
        }
    }
    api_error($code, $message, $status);
}

function validate_api_key($endpoint_group) {
    $key = get_api_key();
    if ($key === '') {
        api_error('missing_api_key', 'This endpoint requires an API key in the X-API-Key header.', 401);
    }
    // Placeholder keys from the example config must never authenticate,
    // even if an enabled entry was left in api_keys.json.
    if (str_starts_with($key, 'REPLACE_ME')) {
        _api_auth_failed('invalid_api_key', 'The supplied API key is not recognised.', 401);
    }
    $keys = load_api_keys();
    if (!isset($keys[$key])) {
        _api_auth_failed('invalid_api_key', 'The supplied API key is not recognised.', 401);
    }
    $key_data = $keys[$key];
    if (empty($key_data['enabled'])) {
        _api_auth_failed('disabled_api_key', 'The supplied API key is disabled.', 403);
    }
    if (!empty($key_data['expires']) && strtotime($key_data['expires']) < time()) {
        _api_auth_failed('expired_api_key', 'The supplied API key has expired.', 403);
    }
    $allowed = $key_data['allowed_endpoints'] ?? [];
    if (!empty($allowed) && !in_array($endpoint_group, $allowed, true)) {
        _api_auth_failed('endpoint_not_allowed', 'This API key is not allowed to use this endpoint group.', 403);
    }
    return $key_data;
}

// --- Input validation helpers ---
function validate_station_id($station_id) {
    if (!preg_match('/^ams\d+$/', $station_id)) {
        api_error('invalid_station_id', 'station_id must match ams\\d+.', 400);
    }
    return $station_id;
}

function validate_camera_num($camera_num) {
    if (!ctype_digit((string)$camera_num)) {
        api_error('invalid_camera_num', 'camera_num must be an integer.', 400);
    }
    return (int)$camera_num;
}

function validate_task_id($task_id, $prefixes = []) {
    $default = '^(?:master_task|task|pass_task|stream|aircraft_task|api_task)_[a-zA-Z0-9_.-]+$';
    $pattern = empty($prefixes) ? $default : '^(?:' . implode('|', $prefixes) . ')_?[a-zA-Z0-9_.-]+$';
    if (!preg_match('/' . $pattern . '/', $task_id)) {
        api_error('invalid_task_id', 'Invalid task ID.', 400);
    }
    return $task_id;
}

function validate_iso_timestamp($ts) {
    if (!preg_match('/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$/', $ts)) {
        api_error('invalid_timestamp', 'Timestamp must be ISO8601 UTC.', 400);
    }
    return $ts;
}

function validate_date($date) {
    if (!preg_match('/^\d{4}-\d{2}-\d{2}$/', $date)) {
        api_error('invalid_date', 'Date must be YYYY-MM-DD.', 400);
    }
    return $date;
}

// --- Path helpers ---
function status_file_path($task_id) {
    return LOCK_DIR . '/' . $task_id . '.json';
}

function write_status_file(string $task_id, array $data) {
    $path = status_file_path($task_id);
    $dir = dirname($path);
    if (!is_dir($dir)) { mkdir($dir, 0775, true); }
    file_put_contents($path, json_encode($data, JSON_THROW_ON_ERROR), LOCK_EX);
}

function shell_quote_array(array $args) {
    return array_map('escapeshellarg', $args);
}

function python_exec() {
    global $PYTHON_EXECUTABLE;
    return $PYTHON_EXECUTABLE;
}

function data_dir() {
    return DATA_DIR;
}
