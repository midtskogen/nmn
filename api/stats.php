<?php
/**
 * API traffic statistics dashboard.
 *
 * Reads nmn/api/query_log.json and nmn/api/abuse_log.json and returns either
 * JSON aggregates or a simple HTML table view (add ?html=1).
 */

require_once __DIR__ . '/api_common.php';
require_once __DIR__ . '/task_queue.php';

$now = time();
$windows = [
    'hour'  => $now - 3600,
    'day'   => $now - 86400,
    'week'  => $now - 7 * 86400,
];

function read_ndjson(string $path): array {
    if (!file_exists($path)) return [];
    $rows = [];
    $handle = fopen($path, 'r');
    if (!$handle) return [];
    while (($line = fgets($handle)) !== false) {
        $line = trim($line);
        if ($line === '') continue;
        $row = json_decode($line, true);
        if (is_array($row)) $rows[] = $row;
    }
    fclose($handle);
    return $rows;
}

function parse_ts(array $row): int {
    return isset($row['ts']) ? strtotime($row['ts']) ?: 0 : 0;
}

function aggregate(array $rows, int $since): array {
    $total = 0;
    $by_ip = [];
    $by_key = [];
    $by_endpoint = [];
    $by_status = [];
    $bytes = 0;
    $quota_hits = 0;

    foreach ($rows as $r) {
        $ts = parse_ts($r);
        if ($ts < $since) continue;
        $total++;
        $ip = $r['ip'] ?? 'unknown';
        $key = $r['key_id'] ?? 'anonymous';
        $ep = $r['endpoint'] ?? 'unknown';
        $st = isset($r['status']) ? (string)$r['status'] : 'unknown';
        $by_ip[$ip] = ($by_ip[$ip] ?? 0) + 1;
        $by_key[$key] = ($by_key[$key] ?? 0) + 1;
        $by_endpoint[$ep] = ($by_endpoint[$ep] ?? 0) + 1;
        $by_status[$st] = ($by_status[$st] ?? 0) + 1;
        $bytes += (int)($r['bytes'] ?? 0);
        if (!empty($r['quota_hit'])) $quota_hits++;
    }

    arsort($by_ip);
    arsort($by_key);
    arsort($by_endpoint);

    return [
        'requests'     => $total,
        'bytes_total'  => $bytes,
        'quota_hits'   => $quota_hits,
        'top_ips'      => array_slice($by_ip, 0, 20, true),
        'top_keys'     => array_slice($by_key, 0, 20, true),
        'top_endpoints'=> array_slice($by_endpoint, 0, 20, true),
        'status_codes' => $by_status,
    ];
}

function abuse_summary(array $rows, int $since): array {
    $counts = [];
    foreach ($rows as $r) {
        if (parse_ts($r) < $since) continue;
        $type = $r['type'] ?? 'unknown';
        $counts[$type] = ($counts[$type] ?? 0) + 1;
    }
    return $counts;
}

$stats = [];
foreach ($windows as $name => $since) {
    $stats[$name] = aggregate(read_ndjson(QUERY_LOG_FILE), $since);
    $stats[$name]['abuse'] = abuse_summary(read_ndjson(ABUSE_LOG_FILE), $since);
}
$stats['queue'] = queue_summary();

$as_html = !empty($_GET['html']);
if (!$as_html) {
    header('Content-Type: application/json');
    echo json_encode($stats, JSON_PRETTY_PRINT);
    exit;
}

// Minimal HTML dashboard.
header('Content-Type: text/html; charset=utf-8');
?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NMN API stats</title>
<style>
body { font-family: system-ui, sans-serif; margin: 2rem; background: #f6f8fa; color: #24292f; }
h1,h2 { color: #1f2328; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; }
.card { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 1rem; }
table { border-collapse: collapse; width: 100%; margin-top: .5rem; }
th,td { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid #eee; }
th { font-weight: 600; }
.summary { font-size: 1.1rem; margin: .5rem 0; }
</style>
</head>
<body>
<h1>NMN API stats</h1>
<?php foreach ($stats as $window => $data): if ($window === 'queue') continue; ?>
<div class="card">
<h2><?= htmlspecialchars(ucfirst($window), ENT_QUOTES, 'UTF-8') ?></h2>
<div class="summary">
  Requests: <strong><?= (int)($data['requests']) ?></strong> |
  Bytes: <strong><?= number_format((int)($data['bytes_total'])) ?></strong> |
  Quota hits: <strong><?= (int)($data['quota_hits']) ?></strong>
</div>
<?php if (!empty($data['abuse'])): ?>
<h3>Abuse events</h3>
<table><tr><th>Type</th><th>Count</th></tr>
<?php foreach ($data['abuse'] as $type => $cnt): ?>
<tr><td><?= htmlspecialchars($type) ?></td><td><?= (int)$cnt ?></td></tr>
<?php endforeach; ?></table>
<?php endif; ?>
<h3>Top IPs</h3>
<table><tr><th>IP</th><th>Requests</th></tr>
<?php foreach (array_slice($data['top_ips'], 0, 10, true) as $ip => $cnt): ?>
<tr><td><?= htmlspecialchars($ip) ?></td><td><?= (int)$cnt ?></td></tr>
<?php endforeach; ?></table>
<h3>Top endpoints</h3>
<table><tr><th>Endpoint</th><th>Requests</th></tr>
<?php foreach (array_slice($data['top_endpoints'], 0, 10, true) as $ep => $cnt): ?>
<tr><td><?= htmlspecialchars($ep) ?></td><td><?= (int)$cnt ?></td></tr>
<?php endforeach; ?></table>
</div>
<?php endforeach; ?>
<div class="card">
<h2>Queue</h2>
<table><tr><th>Status</th><th>Count</th></tr>
<?php foreach ($stats['queue'] as $status => $cnt): ?>
<tr><td><?= htmlspecialchars($status) ?></td><td><?= (int)$cnt ?></td></tr>
<?php endforeach; ?></table>
</div>
</body>
</html>
<?php
