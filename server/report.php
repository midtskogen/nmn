<?php
// Report-generation trigger called by stations after processing an event.
// Security model:
//   - Shared-secret token REQUIRED (X-NMN-Token header, or &token=);
//     the secret lives in /var/www/.ssh/report_token, outside the web root.
//   - station must exist in stations.json and map to a tunnel port in
//     /var/www/.ssh/config - the client-supplied ?port= is ignored, so a
//     ping can never aim a fetch at an arbitrary forwarded port.
//   - dir must be a literal /meteor/camN/amsevents/YYYYMMDD/HHMMSS[_N]
//     path, so rsync can only pull real event directories.
//   - One fetch per dir per hour (dedupe), plus a coarse per-IP rate limit.
//   - Denied/skipped requests are logged to /tmp/report_php.log.

$REPORT_TOKEN = trim((string)@file_get_contents('/var/www/.ssh/report_token'));
if ($REPORT_TOKEN === '') {
    http_response_code(500);
    exit("token not configured\n");
}

header('Content-Type: text/plain; charset=UTF-8');

$dir     = (string)($_GET['dir'] ?? '');
$station = preg_replace('/[^\w]/', '', (string)($_GET['station'] ?? ''));
$port    = preg_replace('/[^\d]/', '', (string)($_GET['port'] ?? ''));
$token   = (string)($_SERVER['HTTP_X_NMN_TOKEN'] ?? $_GET['token'] ?? '');
$ip      = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
$log_file = '/tmp/report_php.log';

// Shared-secret token is REQUIRED (deployed to stations via
// /etc/default/nmn_report_token; see bin/report.py in the repo).
if ($token === '' || !hash_equals($REPORT_TOKEN, $token)) {
    deny(403, 'forbidden');
}

function deny(int $code, string $msg): void {
    global $dir, $station, $ip, $log_file;
    @file_put_contents($log_file,
        "[" . date("Y-m-d H:i:s") . "] DENIED $code $msg station=$station dir=$dir ip=$ip\n",
        FILE_APPEND | LOCK_EX);
    http_response_code($code);
    echo $msg, "\n";
    exit;
}

// Whitelist station names against stations.json and resolve the amsNNN id.
$ams_id = '';
$j = @json_decode((string)@file_get_contents('/var/www/html/data/stations.json'), true);
if (is_array($j)) {
    foreach ($j as $k => $e) {
        if (($e['station']['name'] ?? '') === $station || $k === $station) {
            $ams_id = $k;
            break;
        }
    }
}
if ($ams_id === '') deny(403, 'unknown station');

// Port is pinned server-side: host amsNNN -> Port in the tunnel ssh config.
// The ?port= parameter is ignored so a ping can never aim the fetch at an
// arbitrary forwarded port.
$port = '';
$cur_host = '';
foreach ((array)@file('/var/www/.ssh/config') as $l) {
    if (preg_match('/^\s*host\s+(\S+)/i', $l, $m)) $cur_host = $m[1];
    elseif ($cur_host === $ams_id && preg_match('/^\s*port\s+(\d+)/i', $l, $m))
        $port = $m[1];
}
if ($port === '') deny(403, 'no tunnel for station');

// dir: anchored to the real event layout so nothing else can be pulled.
$clean_dir = preg_replace('/[^a-zA-Z0-9_\/\-]/', '', $dir);
$clean_dir = '/' . trim($clean_dir, '/');
if (!preg_match('#^/meteor/cam\d+/amsevents/\d{8}/\d{6}(_\d+)?$#', $clean_dir))
    deny(400, 'bad dir');

// Per-IP rate limit: 20 pings/hour. Dedupe: same dir once per hour.
$rl_file = '/tmp/report_php_ratelimit.json';
$fp = fopen($rl_file, 'c+');
if ($fp && flock($fp, LOCK_EX)) {
    $b = json_decode((string)stream_get_contents($fp), true) ?: [];
    $now = time();
    $b[$ip] = array_values(array_filter($b[$ip] ?? [], fn($t) => $t > $now - 3600));
    $seen_key = 'dir:' . $station . ':' . $clean_dir;
    $b[$seen_key] = array_values(array_filter($b[$seen_key] ?? [], fn($t) => $t > $now - 3600));
    $dup = count($b[$seen_key]) > 0;
    if (count($b[$ip]) >= 20) { flock($fp, LOCK_UN); fclose($fp); deny(429, 'rate limited'); }
    $b[$ip][] = $now;
    $b[$seen_key][] = $now;
    ftruncate($fp, 0); rewind($fp); fwrite($fp, json_encode($b));
    flock($fp, LOCK_UN); fclose($fp);
    if ($dup) {
        @file_put_contents($log_file,
            "[" . date("Y-m-d H:i:s") . "] SKIPPED duplicate station=$station dir=$clean_dir ip=$ip\n",
            FILE_APPEND | LOCK_EX);
        echo "already queued\n"; exit;
    }
}

$ts = date('Y-m-d H:i:s');
$token_ok = $token !== '' && hash_equals($REPORT_TOKEN, $token);
$log_entry = "[$ts] station=$station port=$port dir=$clean_dir ip=$ip token=" . ($token_ok ? 'ok' : 'missing/bad') . "\n";

$fetch_sh = '/home/httpd/norskmeteornettverk.no/bin/fetch.sh';
if (!is_file($fetch_sh)) $fetch_sh = '/home/steinar/norskmeteornettverk.no/nmn/server/fetch.sh';

$cmd = escapeshellarg($fetch_sh) . ' ' . escapeshellarg($station) . ' ' . escapeshellarg($port) . ' ' . escapeshellarg($clean_dir);
$log_entry .= "[$ts] Executing: $cmd\n";
$logfile = '/tmp/fetch_output_' . $station . '_' . time() . '.log';
shell_exec('nohup ' . $cmd . ' > ' . escapeshellarg($logfile) . ' 2>&1 < /dev/null &');

$log_entry .= "[$ts] Launched\n\n";
file_put_contents($log_file, $log_entry, FILE_APPEND | LOCK_EX);

echo "Report is being generated for station $station, port $port.\n";
