<?php
/**
 * Background GeoIP resolver — called by usage.php via shell_exec.
 * Resolves IPs in $pending_file against ip-api.com and writes
 * results into $cache_file. Runs detached so it never blocks the page.
 *
 * Usage: php geoip_resolve.php <cache_file> <pending_file>
 */
if ($argc < 3) exit(1);

$cache_file   = realpath($argv[1]) ?: $argv[1];
$pending_file = realpath($argv[2]) ?: $argv[2];

// Constrain both files to the same directory as this script (i.e. server/data/cache).
$base_dir = realpath(__DIR__ . '/cache');
if ($base_dir === false || strpos($cache_file, $base_dir . DIRECTORY_SEPARATOR) !== 0 || strpos($pending_file, $base_dir . DIRECTORY_SEPARATOR) !== 0) {
    exit(1);
}

if (!file_exists($pending_file)) exit(0);

$ips = array_filter(array_map('trim', file($pending_file)));
@unlink($pending_file);

if (!$ips) exit(0);

// Load existing cache
$cache = [];
if (file_exists($cache_file)) {
    $data = json_decode(file_get_contents($cache_file), true);
    if (is_array($data)) $cache = $data;
}

$lock = $cache_file . '.lock';
$lf   = fopen($lock, 'c');
if (!flock($lf, LOCK_EX)) exit(1);

// Re-read after acquiring lock (another process may have written)
if (file_exists($cache_file)) {
    $data = json_decode(file_get_contents($cache_file), true);
    if (is_array($data)) $cache = $data;
}

foreach ($ips as $ip) {
    if (isset($cache[$ip])) continue;
    if (!filter_var($ip, FILTER_VALIDATE_IP)) continue;
    $url  = "https://ip-api.com/json/{$ip}?fields=country,countryCode,status";
    $ctx  = stream_context_create(['http' => ['timeout' => 3], 'ssl' => ['verify_peer' => true, 'verify_peer_name' => true]]);
    $json = @file_get_contents($url, false, $ctx);
    if ($json) {
        $d = json_decode($json, true);
        $cache[$ip] = (isset($d['status']) && $d['status'] === 'success')
            ? ['country' => $d['country'], 'cc' => $d['countryCode']]
            : ['country' => '', 'cc' => ''];
    } else {
        $cache[$ip] = ['country' => '', 'cc' => ''];
    }
    usleep(250000); // 250ms — stay within free tier (45 req/min)
}

file_put_contents($cache_file, json_encode($cache), LOCK_EX);
flock($lf, LOCK_UN);
fclose($lf);
// Do NOT unlink the lock file: a process blocked on flock of the unlinked inode
// would hold a lock on a deleted file while a new process locks a fresh inode.
