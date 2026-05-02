<?php
/**
 * Background GeoIP resolver — called by usage.php via shell_exec.
 * Resolves IPs in $pending_file against ip-api.com and writes
 * results into $cache_file. Runs detached so it never blocks the page.
 *
 * Usage: php geoip_resolve.php <cache_file> <pending_file>
 */
if ($argc < 3) exit(1);

$cache_file   = $argv[1];
$pending_file = $argv[2];

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
    $url  = "http://ip-api.com/json/{$ip}?fields=country,countryCode,status";
    $ctx  = stream_context_create(['http' => ['timeout' => 3]]);
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

file_put_contents($cache_file, json_encode($cache, JSON_PRETTY_PRINT));
flock($lf, LOCK_UN);
fclose($lf);
@unlink($lock);
