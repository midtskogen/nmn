<?php
/**
 * API request and abuse logging.
 *
 * Writes compact NDJSON lines to nmn/api/query_log.json and nmn/api/abuse_log.json.
 * Both files are rotated when they exceed the limits configured in
 * etc/api_config.json.
 */

if (!defined('QUERY_LOG_FILE')) {
    define('QUERY_LOG_FILE', __DIR__ . '/query_log.json');
}
if (!defined('ABUSE_LOG_FILE')) {
    define('ABUSE_LOG_FILE', __DIR__ . '/abuse_log.json');
}

function _log_config(): array {
    if (!function_exists('load_api_config')) {
        require_once __DIR__ . '/api_common.php';
    }
    $cfg = load_api_config();
    return $cfg['logging'] ?? [
        'query_log_max_bytes'  => 50 * 1024 * 1024,
        'abuse_log_max_bytes'  => 10 * 1024 * 1024,
    ];
}

function _rotate_log_if_needed(string $file, int $max_bytes) {
    if (file_exists($file) && filesize($file) > $max_bytes) {
        @rename($file, $file . '.old');
    }
}

function _append_ndjson(string $file, array $entry) {
    $dir = dirname($file);
    if (!is_dir($dir)) { mkdir($dir, 0775, true); }
    $line = json_encode($entry, JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR) . "\n";
    file_put_contents($file, $line, FILE_APPEND | LOCK_EX);
}

/**
 * Log an API request.
 */
function log_api_request(array $info) {
    $cfg = _log_config();
    _rotate_log_if_needed(QUERY_LOG_FILE, $cfg['query_log_max_bytes']);
    $entry = array_merge([
        'ts'      => date('Y-m-d H:i:s'),
        'date'    => date('Y-m-d'),
        'ip'      => 'unknown_ip',
        'key_id'  => null,
        'method'  => 'GET',
        'endpoint'=> '',
        'station' => null,
        'status'  => 200,
        'ms'      => 0,
        'bytes'   => 0,
        'quota_hit' => false,
        'error'   => null,
    ], $info);
    _append_ndjson(QUERY_LOG_FILE, $entry);
}

/**
 * Log an abuse/rate-limit/quota event.
 */
function log_abuse_event(array $info) {
    $cfg = _log_config();
    _rotate_log_if_needed(ABUSE_LOG_FILE, $cfg['abuse_log_max_bytes']);
    $entry = array_merge([
        'ts'      => date('Y-m-d H:i:s'),
        'date'    => date('Y-m-d'),
        'ip'      => 'unknown_ip',
        'key_id'  => null,
        'type'    => 'rate_limit',
        'endpoint'=> '',
        'station' => null,
        'reason'  => '',
    ], $info);
    _append_ndjson(ABUSE_LOG_FILE, $entry);
}
