<?php
/**
 * Sliding-window rate limiting for the NMN API.
 *
 * Uses a small JSON file protected by flock.  Each bucket stores the timestamps
 * of the most recent requests; old entries are pruned when the window rolls.
 */

if (!defined('RATE_LIMIT_FILE')) {
    define('RATE_LIMIT_FILE', __DIR__ . '/rate_limit.json');
}

function default_rate_limits(): array {
    return [
        'read_only' => ['requests_per_minute' => 100, 'requests_per_hour' => 1000],
        'stateful'  => ['requests_per_minute' => 20,  'requests_per_hour' => 200],
        // Queued CPU-heavy jobs get a much tighter anonymous budget.
        'predict'   => ['requests_per_minute' => 5,   'requests_per_hour' => 40],
        // Failed API-key authentication attempts per IP.
        'auth_fail' => ['requests_per_minute' => 10,  'requests_per_hour' => 60],
    ];
}

function _rate_limit_config(): array {
    if (!function_exists('load_api_config')) {
        require_once __DIR__ . '/api_common.php';
    }
    $cfg = load_api_config();
    $limits = [];
    foreach (default_rate_limits() as $type => $defaults) {
        $limits[$type] = array_merge($defaults, $cfg['default_rate_limit'][$type] ?? []);
    }
    return $limits;
}

/**
 * Check a sliding-window rate limit.
 *
 * @param string $bucket_key    Identifier for the bucket (IP address or API key).
 * @param string $type          'read_only' or 'stateful'.
 * @param array|null $override  Optional per-bucket override from api_keys.json.
 * @return array ['allowed' => bool, 'retry_after' => int]
 */
function check_rate_limit(string $bucket_key, string $type, ?array $override = null): array {
    $limits = _rate_limit_config()[$type] ?? _rate_limit_config()['read_only'];
    if ($override !== null) {
        $limits = array_merge($limits, $override);
    }
    $now = time();
    $max_minute = (int) $limits['requests_per_minute'];
    $max_hour   = (int) $limits['requests_per_hour'];

    $file = RATE_LIMIT_FILE;
    $dir = dirname($file);
    if (!is_dir($dir)) { mkdir($dir, 0775, true); }

    $fp = fopen($file, 'c');
    if (!$fp || !flock($fp, LOCK_EX)) {
        // If we cannot lock, fail open (allow the request) but log is impossible here.
        return ['allowed' => true, 'retry_after' => 0];
    }

    $data = [];
    if (filesize($file) > 0) {
        $raw = fread($fp, filesize($file));
        $decoded = json_decode($raw, true);
        if (is_array($decoded)) $data = $decoded;
    }

    if (!isset($data[$bucket_key])) {
        $data[$bucket_key] = ['minute' => [], 'hour' => []];
    }

    // Prune old entries.
    $data[$bucket_key]['minute'] = array_values(array_filter($data[$bucket_key]['minute'], fn($t) => $t > $now - 60));
    $data[$bucket_key]['hour']   = array_values(array_filter($data[$bucket_key]['hour'],   fn($t) => $t > $now - 3600));

    $minute_count = count($data[$bucket_key]['minute']);
    $hour_count   = count($data[$bucket_key]['hour']);

    $allowed = true;
    $retry_after = 0;
    if ($minute_count >= $max_minute) {
        $allowed = false;
        $retry_after = max($retry_after, 60 - ($now - min($data[$bucket_key]['minute'])));
    }
    if ($hour_count >= $max_hour) {
        $allowed = false;
        $retry_after = max($retry_after, 3600 - ($now - min($data[$bucket_key]['hour'])));
    }

    if ($allowed) {
        $data[$bucket_key]['minute'][] = $now;
        $data[$bucket_key]['hour'][]   = $now;
    }

    ftruncate($fp, 0);
    rewind($fp);
    fwrite($fp, json_encode($data, JSON_THROW_ON_ERROR));
    fflush($fp);
    flock($fp, LOCK_UN);
    fclose($fp);

    return ['allowed' => $allowed, 'retry_after' => (int) ceil($retry_after)];
}
