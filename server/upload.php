<?php
// Station event push endpoint: the station POSTs its event directory as a
// tar.gz body; we store it in quarantine and hand it to receive_upload.py,
// which validates every member, copies it into the event tree and runs the
// same merge+process pipeline as an rsync pull.
//
// Security model (mirrors report.php):
//   - ssh-keygen signature (X-NMN-Sig) over "<dir>\n<body>" REQUIRED.
//   - station must exist in stations.json (whitelist, not just \w).
//   - dir must be a literal /meteor/camN/amsevents/YYYYMMDD/HHMMSS[_N] path.
//   - Upload size is capped while streaming; dedupe + per-IP rate limit.
//   - The body lands in quarantine OUTSIDE the event tree; only after
//     receive_upload.py's member validation does anything reach meteor/.

$allowed_signers = '/var/www/.ssh/allowed_signers';   // station ssh pubkey(s)

header('Content-Type: text/plain; charset=UTF-8');

$dir     = (string)($_GET['dir'] ?? '');
$station = preg_replace('/[^\w]/', '', (string)($_GET['station'] ?? ''));
$sig_b64 = (string)($_SERVER['HTTP_X_NMN_SIG'] ?? '');
$ip      = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
$log_file = '/tmp/upload_php.log';

function deny(int $code, string $msg): void {
    global $dir, $station, $ip, $log_file;
    @file_put_contents($log_file,
        "[" . date("Y-m-d H:i:s") . "] DENIED $code $msg station=$station dir=$dir ip=$ip\n",
        FILE_APPEND | LOCK_EX);
    http_response_code($code);
    echo $msg, "\n";
    exit;
}

// Authentication: an ssh-keygen signature over "<dir>\n<body>" made with
// the station's private key (the key the reverse tunnel uses - it never
// leaves the station), verified against allowed_signers.  Signature only;
// there is no shared-secret fallback on this endpoint.
$auth = '';

// Whitelist station names against stations.json.
$stations_ok = false;
$j = @json_decode((string)@file_get_contents('/var/www/html/data/stations.json'), true);
if (is_array($j)) {
    $stations_ok = isset($j[$station])
        || in_array($station, array_map(fn($e) => $e['station']['name'] ?? '', $j), true);
}
if (!$stations_ok) deny(403, 'unknown station');

// dir: anchored to the real event layout.
$clean_dir = preg_replace('/[^a-zA-Z0-9_\/\-]/', '', $dir);
$clean_dir = '/' . trim($clean_dir, '/');
if (!preg_match('#^/meteor/cam\d+/amsevents/\d{8}/\d{6}(_\d+)?$#', $clean_dir))
    deny(400, 'bad dir');

// Per-IP rate limit (30/hr) + dedupe (same dir once per hour).
$rl_file = '/tmp/upload_php_ratelimit.json';
$fp = fopen($rl_file, 'c+');
if ($fp && flock($fp, LOCK_EX)) {
    $b = json_decode((string)stream_get_contents($fp), true) ?: [];
    $now = time();
    $b[$ip] = array_values(array_filter($b[$ip] ?? [], fn($t) => $t > $now - 3600));
    $seen_key = 'dir:' . $station . ':' . $clean_dir;
    $b[$seen_key] = array_values(array_filter($b[$seen_key] ?? [], fn($t) => $t > $now - 3600));
    $dup = count($b[$seen_key]) > 0;
    if (count($b[$ip]) >= 30) { flock($fp, LOCK_UN); fclose($fp); deny(429, 'rate limited'); }
    $b[$ip][] = $now;
    // Note: the dedupe mark is added only after the upload is stored —
    // a failed attempt must be retryable.
    $tmp = $rl_file . '.tmp';
    file_put_contents($tmp, json_encode($b));
    rename($tmp, $rl_file);
    flock($fp, LOCK_UN);
    fclose($fp);
    if ($dup) {
        @file_put_contents($log_file,
            "[" . date("Y-m-d H:i:s") . "] SKIPPED duplicate station=$station dir=$clean_dir ip=$ip\n",
            FILE_APPEND | LOCK_EX);
        echo "already queued\n"; exit;
    }
}

// Stream the request body into quarantine with a hard size cap.
$quarantine = '/var/www/incoming/' . $station;
if (!is_dir($quarantine) && !@mkdir($quarantine, 0750, true))
    deny(500, 'quarantine unavailable');
$stamp = preg_replace('/[^0-9_]/', '', str_replace('/', '_', $clean_dir));
$archive = $quarantine . '/' . $stamp . '_' . bin2hex(random_bytes(4)) . '.tar.gz';

$in  = fopen('php://input', 'rb');
$out = fopen($archive, 'wb');
$MAX = 400 * 1024 * 1024;
$written = 0;
if ($in && $out) {
    while (!feof($in)) {
        $chunk = fread($in, 1 << 20);
        if ($chunk === false) break;
        $written += strlen($chunk);
        if ($written > $MAX) break;
        fwrite($out, $chunk);
    }
}
if ($in) fclose($in);
if ($out) fclose($out);
if ($written === 0) { @unlink($archive); deny(400, 'empty body'); }
if ($written > $MAX) { @unlink($archive); deny(413, 'too large'); }

// Authenticate: signature over "<dir>\n<body>".
if ($sig_b64 !== '') {
    $sigfile = $archive . '.sig';
    $payload = $archive . '.payload';
    file_put_contents($sigfile, base64_decode($sig_b64));
    $pf = fopen($payload, 'wb');
    fwrite($pf, $clean_dir . "\n");
    $af = fopen($archive, 'rb');
    stream_copy_to_stream($af, $pf);
    fclose($af); fclose($pf);
    $v = 'ssh-keygen -Y verify -f ' . escapeshellarg($allowed_signers)
       . ' -I nmn-station -n nmn-upload -s ' . escapeshellarg($sigfile)
       . ' < ' . escapeshellarg($payload) . ' 2>/dev/null';
    exec($v, $vo, $vrc);
    @unlink($sigfile); @unlink($payload);
    if ($vrc === 0) $auth = 'sig';
}
if ($auth === '') { @unlink($archive); deny(403, 'forbidden'); }

// Upload stored — now mark the dir as seen for dedupe.
if ($fp = fopen($rl_file, 'c+')) {
    if (flock($fp, LOCK_EX)) {
        $b = json_decode((string)stream_get_contents($fp), true) ?: [];
        $b[$seen_key][] = time();
        $tmp = $rl_file . '.tmp';
        file_put_contents($tmp, json_encode($b));
        rename($tmp, $rl_file);
        flock($fp, LOCK_UN);
    }
    fclose($fp);
}

// Hand off to the receiver (validates archive members, promotes, processes).
$test_dir = '/home/steinar/norskmeteornettverk.no/nmn/server';
$server_dir = '/home/httpd/norskmeteornettverk.no/nmn/server';
$script_dir = is_dir($test_dir) ? $test_dir : $server_dir;
$rec = $script_dir . '/receive_upload.py';
if (!is_file($rec)) { @unlink($archive); deny(500, 'receiver missing'); }

$ulog = '/tmp/upload_output_' . $station . '_' . time() . '.log';
$cmd = sprintf(
    'nohup python3 %s %s %s %s > %s 2>&1 &',
    escapeshellarg($rec),
    escapeshellarg($archive),
    escapeshellarg($station),
    escapeshellarg($clean_dir),
    escapeshellarg($ulog));
exec($cmd);

@file_put_contents($log_file,
    "[" . date("Y-m-d H:i:s") . "] RECEIVED auth=$auth station=$station dir=$clean_dir bytes=$written ip=$ip archive=$archive\n",
    FILE_APPEND | LOCK_EX);
echo "received\n";
