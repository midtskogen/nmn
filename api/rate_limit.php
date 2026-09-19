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
 * Normalize a bucket key so that IPv6 addresses share a /64 bucket.
 * Without this, an attacker rotating through a single /64 (18 quintillion
 * addresses) trivially bypasses every per-IP limit and grows the state file
 * without bound.  Non-IP keys (e.g. API key ids) pass through unchanged.
 */
function _rate_bucket_key(string $bucket_key): string {
    if (filter_var($bucket_key, FILTER_VALIDATE_IP, FILTER_FLAG_IPV6)) {
        $packed = inet_pton($bucket_key);
        if ($packed !== false && strlen($packed) === 16) {
            // First 8 bytes = /64 network prefix.
            return 'v6:' . bin2hex(substr($packed, 0, 8)) . '/64';
        }
    }
    if (filter_var($bucket_key, FILTER_VALIDATE_IP, FILTER_FLAG_IPV4)) {
        return 'v4:' . $bucket_key;
    }
    return $bucket_key;
}

// Upper bound on distinct buckets retained in the state file.  Beyond this,
// the least-recently-active buckets are evicted — this bounds the file size
// (and therefore the flock-serialized read/rewrite cost per request).
define('RATE_LIMIT_MAX_BUCKETS', 5000);

/**
 * Check a sliding-window rate limit.
 *
 * @param string $bucket_key    Identifier for the bucket (IP address or API key).
 * @param string $type          'read_only' or 'stateful'.
 * @param array|null $override  Optional per-bucket override from api_keys.json.
 * @return array ['allowed' => bool, 'retry_after' => int]
 */
function check_rate_limit(string $bucket_key, string $type, ?array $override = null): array {
    $bucket_key = _rate_bucket_key($bucket_key);
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
        if ($fp) fclose($fp);
        // Failing open here silently disables all rate limiting (including
        // the auth-failure brute-force throttle).  Deny the security-
        // sensitive bucket types; for the rest, allow but make the failure
        // visible in the error log.
        error_log("NMN rate_limit: cannot lock $file — bucket type '$type'");
        if (in_array($type, ['auth_fail', 'predict', 'stateful'], true)) {
            return ['allowed' => false, 'retry_after' => 60];
        }
        return ['allowed' => true, 'retry_after' => 0];
    }

    $data = [];
    if (filesize($file) > 0) {
        $raw = fread($fp, filesize($file));
        $decoded = json_decode($raw, true);
        if (is_array($decoded)) $data = $decoded;
    }

    // Evict buckets with no activity inside either window, then cap the
    // total bucket count so the file cannot grow without bound (an attacker
    // rotating IPs must not be able to bloat a file every request rewrites
    // under an exclusive lock).
    foreach ($data as $k => $b) {
        $minute = array_filter($b['minute'] ?? [], fn($t) => $t > $now - 60);
        $hour   = array_filter($b['hour']   ?? [], fn($t) => $t > $now - 3600);
        if (empty($minute) && empty($hour) && $k !== $bucket_key) {
            unset($data[$k]);
        }
    }
    if (count($data) >= RATE_LIMIT_MAX_BUCKETS && !isset($data[$bucket_key])) {
        $last_seen = [];
        foreach ($data as $k => $b) {
            $last_seen[$k] = max(array_merge($b['minute'] ?? [], $b['hour'] ?? [], [0]));
        }
        asort($last_seen);
        foreach (array_slice(array_keys($last_seen), 0, count($data) - RATE_LIMIT_MAX_BUCKETS + 1) as $k) {
            unset($data[$k]);
        }
    }

    if (!isset($data[$bucket_key])) {
        $data[$bucket_key] = ['minute' => [], 'hour' => []];
    }

    // Prune old entries.  Buckets may come from an older/hand-edited file
    // that lacks one of the keys — default to [] rather than TypeError.
    $data[$bucket_key]['minute'] = array_values(array_filter($data[$bucket_key]['minute'] ?? [], fn($t) => $t > $now - 60));
    $data[$bucket_key]['hour']   = array_values(array_filter($data[$bucket_key]['hour']   ?? [], fn($t) => $t > $now - 3600));

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
