<?php
$BASE_DIR = dirname($_SERVER['SCRIPT_FILENAME']);

// --- Language ---
$supported_langs = ['nb_NO', 'en_GB'];
$lang_code = 'nb_NO';
if (isset($_COOKIE['lang']) && in_array($_COOKIE['lang'], $supported_langs)) {
    $lang_code = $_COOKIE['lang'];
} elseif (isset($_SERVER['HTTP_ACCEPT_LANGUAGE'])) {
    $bl = str_replace('-', '_', substr($_SERVER['HTTP_ACCEPT_LANGUAGE'], 0, 5));
    if (in_array($bl, $supported_langs)) $lang_code = $bl;
    else {
        $short = substr($bl, 0, 2);
        foreach ($supported_langs as $s) { if (substr($s, 0, 2) === $short) { $lang_code = $s; break; } }
    }
}
$lang_file = $BASE_DIR . '/lang/' . $lang_code . '.json';
$lang = file_exists($lang_file) ? (json_decode(file_get_contents($lang_file), true) ?: []) : [];
function t($key, $lang) { return htmlspecialchars($lang[$key] ?? $key); }

// Norwegian country name translations (English → Norsk)
$COUNTRY_NO = [
    'Afghanistan' => 'Afghanistan', 'Albania' => 'Albania', 'Algeria' => 'Algerie',
    'Andorra' => 'Andorra', 'Angola' => 'Angola', 'Argentina' => 'Argentina',
    'Armenia' => 'Armenia', 'Australia' => 'Australia', 'Austria' => 'Østerrike',
    'Azerbaijan' => 'Aserbajdsjan', 'Bahrain' => 'Bahrain', 'Bangladesh' => 'Bangladesh',
    'Belarus' => 'Hviterussland', 'Belgium' => 'Belgia', 'Bolivia' => 'Bolivia',
    'Bosnia and Herzegovina' => 'Bosnia-Hercegovina', 'Brazil' => 'Brasil',
    'Bulgaria' => 'Bulgaria', 'Canada' => 'Canada', 'Chile' => 'Chile',
    'China' => 'Kina', 'Colombia' => 'Colombia', 'Croatia' => 'Kroatia',
    'Cyprus' => 'Kypros', 'Czech Republic' => 'Tsjekkia', 'Czechia' => 'Tsjekkia',
    'Denmark' => 'Danmark', 'Ecuador' => 'Ecuador', 'Egypt' => 'Egypt',
    'Estonia' => 'Estland', 'Ethiopia' => 'Etiopia', 'Finland' => 'Finland',
    'France' => 'Frankrike', 'Georgia' => 'Georgia', 'Germany' => 'Tyskland',
    'Ghana' => 'Ghana', 'Greece' => 'Hellas', 'Hungary' => 'Ungarn',
    'Iceland' => 'Island', 'India' => 'India', 'Indonesia' => 'Indonesia',
    'Iran' => 'Iran', 'Iraq' => 'Irak', 'Ireland' => 'Irland',
    'Israel' => 'Israel', 'Italy' => 'Italia', 'Japan' => 'Japan',
    'Jordan' => 'Jordan', 'Kazakhstan' => 'Kasakhstan', 'Kenya' => 'Kenya',
    'Kosovo' => 'Kosovo', 'Kuwait' => 'Kuwait', 'Latvia' => 'Latvia',
    'Lebanon' => 'Libanon', 'Libya' => 'Libya', 'Liechtenstein' => 'Liechtenstein',
    'Lithuania' => 'Litauen', 'Luxembourg' => 'Luxemburg', 'Malaysia' => 'Malaysia',
    'Malta' => 'Malta', 'Mexico' => 'Mexico', 'Moldova' => 'Moldova',
    'Monaco' => 'Monaco', 'Montenegro' => 'Montenegro', 'Morocco' => 'Marokko',
    'Netherlands' => 'Nederland', 'New Zealand' => 'New Zealand',
    'Nigeria' => 'Nigeria', 'North Macedonia' => 'Nord-Makedonia',
    'Norway' => 'Norge', 'Pakistan' => 'Pakistan', 'Palestine' => 'Palestina',
    'Peru' => 'Peru', 'Philippines' => 'Filippinene', 'Poland' => 'Polen',
    'Portugal' => 'Portugal', 'Qatar' => 'Qatar', 'Romania' => 'Romania',
    'Russia' => 'Russland', 'Saudi Arabia' => 'Saudi-Arabia',
    'Serbia' => 'Serbia', 'Singapore' => 'Singapore', 'Slovakia' => 'Slovakia',
    'Slovenia' => 'Slovenia', 'South Africa' => 'Sør-Afrika',
    'South Korea' => 'Sør-Korea', 'Spain' => 'Spania', 'Sweden' => 'Sverige',
    'Switzerland' => 'Sveits', 'Syria' => 'Syria', 'Taiwan' => 'Taiwan',
    'Thailand' => 'Thailand', 'Tunisia' => 'Tunisia', 'Turkey' => 'Tyrkia',
    'Ukraine' => 'Ukraina', 'United Arab Emirates' => 'De forente arabiske emirater',
    'United Kingdom' => 'Storbritannia', 'United States' => 'USA',
    'Uruguay' => 'Uruguay', 'Uzbekistan' => 'Usbekistan',
    'Venezuela' => 'Venezuela', 'Vietnam' => 'Vietnam', 'Yemen' => 'Jemen',
];

$stream_file  = $BASE_DIR . '/stream_time_tracker.json';
$quota_file   = $BASE_DIR . '/quota_tracker.json';
$access_file  = $BASE_DIR . '/access_log.json';
$geoip_cache  = $BASE_DIR . '/cache/geoip_cache.json';

function read_json($path) {
    if (!file_exists($path)) return [];
    $data = json_decode(file_get_contents($path), true);
    return is_array($data) ? $data : [];
}

function read_ndjson($path) {
    if (!file_exists($path)) return [];
    $rows = [];
    foreach (file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) as $line) {
        $r = json_decode($line, true);
        if (is_array($r)) $rows[] = $r;
    }
    return $rows;
}

// --- Server-side GeoIP lookup with persistent cache ---
// Returns cached data immediately. Spawns a background process to resolve
// any unknown IPs so the page never blocks on network calls.
function resolve_geoip($ips, $cache_file) {
    $cache = read_json($cache_file);
    $unknown = array_filter($ips, fn($ip) => !isset($cache[$ip]));
    if ($unknown) {
        // Write unknown IPs to a temp file and resolve in the background
        $tmp = $cache_file . '.pending';
        @file_put_contents($tmp, implode("\n", $unknown));
        $script = __DIR__ . '/geoip_resolve.php';
        @shell_exec("php {$script} " . escapeshellarg($cache_file) . " " . escapeshellarg($tmp) . " > /dev/null 2>&1 &");
    }
    return $cache;
}

// --- Time range filter ---
$range_options = ['1m' => 30, '6m' => 183, '1y' => 365, '3y' => 1095];
$range = 'cookie';
if (isset($_GET['range']) && isset($range_options[$_GET['range']])) {
    $range = $_GET['range'];
    setcookie('usage_range', $range, time() + 86400 * 365, '/');
} elseif (isset($_COOKIE['usage_range']) && isset($range_options[$_COOKIE['usage_range']])) {
    $range = $_COOKIE['usage_range'];
} else {
    $range = '6m';
}
$cutoff_date = date('Y-m-d', strtotime('-' . $range_options[$range] . ' days'));

$stream_data = read_json($stream_file);
$quota_data  = read_json($quota_file);
$access_data = read_ndjson($access_file);

// Apply date cutoff
foreach (array_keys($stream_data) as $d) { if ($d < $cutoff_date) unset($stream_data[$d]); }
foreach (array_keys($quota_data)  as $d) { if ($d < $cutoff_date) unset($quota_data[$d]); }
$access_data = array_filter($access_data, fn($r) => ($r['date'] ?? '') >= $cutoff_date);

// --- Aggregate stream data ---
// Result: $stream_by_day[date][ip][station] = {lowres, hires}
$stream_by_day = [];
foreach ($stream_data as $date => $ips) {
    foreach ($ips as $ip => $stations) {
        // Old format: top-level keys are not station ids but resolution keys
        if (isset($stations['total_lowres_seconds']) || isset($stations['total_hires_seconds'])) {
            // Skip malformed legacy entries without a station key
            continue;
        }
        foreach ($stations as $station => $usage) {
            if (!is_array($usage)) continue;
            $stream_by_day[$date][$ip][$station] = [
                'lowres' => $usage['total_lowres_seconds'] ?? 0,
                'hires'  => $usage['total_hires_seconds']  ?? 0,
            ];
        }
    }
}

// --- Aggregate download data ---
// Result: $quota_by_day[date][ip][station] = bytes
$quota_by_day = [];
foreach ($quota_data as $date => $stations) {
    foreach ($stations as $station => $usage) {
        if (is_int($usage)) {
            // Very old format — no per-IP breakdown
            $quota_by_day[$date]['(unknown)'][$station] = ($quota_by_day[$date]['(unknown)'][$station] ?? 0) + $usage;
        } elseif (is_array($usage) && isset($usage['sites'])) {
            foreach ($usage['sites'] as $ip => $bytes) {
                $quota_by_day[$date][$ip][$station] = ($quota_by_day[$date][$ip][$station] ?? 0) + $bytes;
            }
        }
    }
}

// --- Collect all unique IPs across both trackers ---
$all_ips = [];
foreach ($stream_by_day as $date => $ips) {
    foreach ($ips as $ip => $_) $all_ips[$ip] = true;
}
foreach ($quota_by_day as $date => $ips) {
    foreach ($ips as $ip => $_) $all_ips[$ip] = true;
}
$all_ips = array_keys($all_ips);

// --- Collect all dates ---
$all_dates = array_unique(array_merge(array_keys($stream_by_day), array_keys($quota_by_day)));
rsort($all_dates);

// --- Per-day summary: total stream seconds and download bytes ---
$daily_summary = [];
foreach ($all_dates as $date) {
    $total_stream_s = 0;
    $total_bytes    = 0;
    $unique_ips     = [];
    foreach ($stream_by_day[$date] ?? [] as $ip => $stations) {
        $unique_ips[$ip] = true;
        foreach ($stations as $s) {
            $total_stream_s += $s['lowres'] + $s['hires'];
        }
    }
    foreach ($quota_by_day[$date] ?? [] as $ip => $stations) {
        $unique_ips[$ip] = true;
        foreach ($stations as $b) $total_bytes += $b;
    }
    $daily_summary[$date] = [
        'stream_s'   => $total_stream_s,
        'bytes'      => $total_bytes,
        'unique_ips' => count($unique_ips),
    ];
}

// --- Per-station totals across all time ---
$station_totals = [];
foreach ($stream_by_day as $date => $ips) {
    foreach ($ips as $ip => $stations) {
        foreach ($stations as $station => $s) {
            $station_totals[$station]['stream_s'] = ($station_totals[$station]['stream_s'] ?? 0) + $s['lowres'] + $s['hires'];
        }
    }
}
foreach ($quota_by_day as $date => $ips) {
    foreach ($ips as $ip => $stations) {
        foreach ($stations as $station => $bytes) {
            $station_totals[$station]['bytes'] = ($station_totals[$station]['bytes'] ?? 0) + $bytes;
        }
    }
}
arsort($station_totals); // sort by first value — loosely by stream usage

// --- Top IPs (all-time stream seconds) ---
$ip_totals = [];
foreach ($stream_by_day as $date => $ips) {
    foreach ($ips as $ip => $stations) {
        foreach ($stations as $s) {
            $ip_totals[$ip] = ($ip_totals[$ip] ?? 0) + $s['lowres'] + $s['hires'];
        }
    }
}
arsort($ip_totals);
$top_ips = array_slice($ip_totals, 0, 50, true);

// --- Per-IP download bytes (all-time) ---
$ip_bytes = [];
foreach ($quota_by_day as $date => $ips) {
    foreach ($ips as $ip => $stations) {
        foreach ($stations as $bytes) {
            $ip_bytes[$ip] = ($ip_bytes[$ip] ?? 0) + $bytes;
        }
    }
}
// Merge any IPs that only appear in quota (not streaming) into top_ips
foreach ($ip_bytes as $ip => $bytes) {
    if (!isset($top_ips[$ip])) $top_ips[$ip] = 0;
}

// Resolve countries server-side (cached)
$geoip = resolve_geoip(array_keys($top_ips), $geoip_cache);

// --- Access log summary (last 7 days) ---
$access_summary = [];
$access_cutoff  = date('Y-m-d', strtotime('-7 days'));
foreach ($access_data as $entry) {
    if (!isset($entry['date']) || $entry['date'] < $access_cutoff) continue;
    $action = $entry['action'] ?? 'unknown';
    $access_summary[$action] = ($access_summary[$action] ?? 0) + 1;
}
arsort($access_summary);

function utf8_codepoint($cp) {
    // Encode a Unicode codepoint as a UTF-8 byte string (supports full 4-byte range)
    if ($cp <= 0x7F)     return chr($cp);
    if ($cp <= 0x7FF)    return chr(0xC0|($cp>>6))   . chr(0x80|($cp&0x3F));
    if ($cp <= 0xFFFF)   return chr(0xE0|($cp>>12))  . chr(0x80|(($cp>>6)&0x3F))  . chr(0x80|($cp&0x3F));
    return               chr(0xF0|($cp>>18))  . chr(0x80|(($cp>>12)&0x3F)) . chr(0x80|(($cp>>6)&0x3F)) . chr(0x80|($cp&0x3F));
}

function fmt_duration($seconds, $lang_code = 'nb_NO') {
    $s = (int)$seconds;
    $u_h = $lang_code === 'nb_NO' ? 't' : 'h';
    $u_m = $lang_code === 'nb_NO' ? 'min' : 'm';
    if ($s < 60) return $s . 's';
    if ($s < 3600) return sprintf('%d%s %ds', intdiv($s,60), $u_m, $s%60);
    return sprintf('%d%s %d%s', intdiv($s,3600), $u_h, intdiv($s%3600,60), $u_m);
}

function fmt_bytes($bytes) {
    $b = (float)$bytes;
    if ($b < 1024) return round($b) . ' B';
    if ($b < 1048576) return round($b/1024,1) . ' KB';
    if ($b < 1073741824) return round($b/1048576,1) . ' MB';
    return round($b/1073741824,2) . ' GB';
}
?>
<!DOCTYPE html>
<html lang="<?= htmlspecialchars(str_replace('_', '-', $lang_code)) ?>">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title><?= t('usage_title', $lang) ?> — Norsk Meteornettverk</title>
<link rel="stylesheet" href="../css/bootstrap.min.css">
<style>
  body { background: #0d1117; color: #e6edf3; font-family: system-ui, sans-serif; }
  h1, h2, h3 { color: #58a6ff; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; }
  .table { color: #e6edf3; }
  .table thead th { color: #8b949e; border-color: #30363d; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }
  .table td, .table th { border-color: #21262d; vertical-align: middle; }
  .table tbody tr:hover { background: #1c2128; }
  .badge-station { background: #1f6feb; color: #fff; font-size: .75rem; padding: 2px 7px; border-radius: 4px; }
  .bar-wrap { background: #21262d; border-radius: 4px; height: 8px; min-width: 60px; }
  .bar-fill  { background: #1f6feb; border-radius: 4px; height: 8px; }
  .stat-num { font-size: 1.6rem; font-weight: 700; color: #58a6ff; }
  .stat-label { font-size: .8rem; color: #8b949e; }
  .nav-tabs .nav-link { color: #8b949e; border-color: transparent; }
  .nav-tabs .nav-link.active { color: #e6edf3; background: #161b22; border-color: #30363d #30363d #161b22; }
  .flag { font-size: 1.1em; }
  canvas { max-height: 260px; }
  .table-sm td, .table-sm th { padding: .3rem .6rem; font-size: .85rem; }
  #ip-table td:first-child { font-family: monospace; font-size: .82rem; }
  .country-cell { white-space: nowrap; }
</style>
</head>
<body class="p-3">
<div class="container-fluid" style="max-width:1200px">

<div class="d-flex align-items-center mb-4 gap-3">
  <img src="../nmn.png" height="40" alt="NMN">
  <div>
    <h1 class="mb-0" style="font-size:1.5rem"><?= t('usage_title', $lang) ?></h1>
    <div class="text-muted" style="font-size:.85rem"><?= $lang['usage_subtitle'] ?? 'Norsk Meteornettverk' ?></div>
  </div>
  <div class="ms-auto d-flex align-items-center gap-2 flex-wrap">
    <?php foreach (['1m'=>'usage_range_1m','6m'=>'usage_range_6m','1y'=>'usage_range_1y','3y'=>'usage_range_3y'] as $r => $key): ?>
      <a href="?range=<?= $r ?>" class="btn btn-sm <?= $range===$r ? 'btn-primary' : 'btn-outline-secondary' ?>" style="font-size:.8rem"><?= t($key,$lang) ?></a>
    <?php endforeach; ?>
    <span class="text-muted" style="font-size:.78rem"><?= t('usage_generated', $lang) ?> <?= htmlspecialchars(date('Y-m-d H:i:s T')) ?></span>
  </div>
</div>

<!-- Summary cards -->
<?php
$total_stream_s_all = array_sum(array_column($daily_summary, 'stream_s'));
$total_bytes_all    = array_sum(array_column($daily_summary, 'bytes'));
$total_days         = count($daily_summary);
$total_ips          = count($all_ips);
?>
<div class="row g-3 mb-4">
  <div class="col-6 col-md-3"><div class="card p-3 text-center">
    <div class="stat-num"><?= $total_days ?></div>
    <div class="stat-label"><?= t('usage_days_active', $lang) ?></div>
  </div></div>
  <div class="col-6 col-md-3"><div class="card p-3 text-center">
    <div class="stat-num"><?= $total_ips ?></div>
    <div class="stat-label"><?= t('usage_unique_ips', $lang) ?></div>
  </div></div>
  <div class="col-6 col-md-3"><div class="card p-3 text-center">
    <div class="stat-num"><?= fmt_duration($total_stream_s_all, $lang_code) ?></div>
    <div class="stat-label"><?= t('usage_total_stream', $lang) ?></div>
  </div></div>
  <div class="col-6 col-md-3"><div class="card p-3 text-center">
    <div class="stat-num"><?= fmt_bytes($total_bytes_all) ?></div>
    <div class="stat-label"><?= t('usage_total_downloaded', $lang) ?></div>
  </div></div>
</div>

<!-- Tabs -->
<ul class="nav nav-tabs mb-3" id="mainTabs">
  <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-daily"><?= t('usage_tab_daily', $lang) ?></button></li>
  <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-stations"><?= t('usage_tab_stations', $lang) ?></button></li>
  <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-ips"><?= t('usage_tab_ips', $lang) ?></button></li>
  <?php if ($access_summary): ?><li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-access"><?= t('usage_tab_actions', $lang) ?></button></li><?php endif; ?>
  <li class="nav-item ms-auto"><button class="nav-link" onclick="document.cookie='lang=nb_NO;path=/;max-age=31536000';location.reload()" title="Norsk">🇳🇴</button></li>
  <li class="nav-item"><button class="nav-link" onclick="document.cookie='lang=en_GB;path=/;max-age=31536000';location.reload()" title="English">🇬🇧</button></li>
</ul>

<div class="tab-content">

<!-- DAILY TAB -->
<div class="tab-pane fade show active" id="tab-daily">
  <div class="card p-3 mb-3">
    <canvas id="chartStream"></canvas>
  </div>
  <div class="card p-3">
    <table class="table table-sm table-hover mb-0">
      <thead><tr>
        <th><?= t('usage_col_date', $lang) ?></th><th><?= t('usage_col_unique_ips', $lang) ?></th><th><?= t('usage_col_stream', $lang) ?></th><th><?= t('usage_col_downloaded', $lang) ?></th><th style="min-width:120px"><?= t('usage_col_stream', $lang) ?></th>
      </tr></thead>
      <tbody>
      <?php
      $max_s = max(1, max(array_column($daily_summary, 'stream_s')));
      foreach ($daily_summary as $date => $row):
          $pct = min(100, round($row['stream_s'] / $max_s * 100));
      ?>
      <tr>
        <td><?= htmlspecialchars($date) ?></td>
        <td><?= $row['unique_ips'] ?></td>
        <td><?= fmt_duration($row['stream_s'], $lang_code) ?></td>
        <td><?= fmt_bytes($row['bytes']) ?></td>
        <td><div class="bar-wrap"><div class="bar-fill" style="width:<?= $pct ?>%"></div></div></td>
      </tr>
      <?php endforeach; ?>
      </tbody>
    </table>
  </div>
</div>

<!-- STATIONS TAB -->
<div class="tab-pane fade" id="tab-stations">
  <div class="card p-3">
    <table class="table table-sm table-hover mb-0">
      <thead><tr>
        <th><?= t('usage_col_station', $lang) ?></th><th><?= t('usage_col_total_stream', $lang) ?></th><th><?= t('usage_col_total_downloaded', $lang) ?></th><th style="min-width:140px"><?= t('usage_col_stream', $lang) ?></th>
      </tr></thead>
      <tbody>
      <?php
      $max_st = max(1, max(array_map(fn($v) => $v['stream_s'] ?? 0, $station_totals)));
      foreach ($station_totals as $station => $totals):
          $pct = min(100, round(($totals['stream_s'] ?? 0) / $max_st * 100));
      ?>
      <tr>
        <td><span class="badge-station"><?= htmlspecialchars($station) ?></span></td>
        <td><?= fmt_duration($totals['stream_s'] ?? 0, $lang_code) ?></td>
        <td><?= fmt_bytes($totals['bytes'] ?? 0) ?></td>
        <td><div class="bar-wrap"><div class="bar-fill" style="width:<?= $pct ?>%"></div></div></td>
      </tr>
      <?php endforeach; ?>
      </tbody>
    </table>
  </div>

  <!-- Per-station per-day breakdown -->
  <?php
  // Build per-station daily totals
  $station_daily = [];
  foreach ($stream_by_day as $date => $ips) {
      foreach ($ips as $ip => $stations) {
          foreach ($stations as $station => $s) {
              $station_daily[$station][$date] = ($station_daily[$station][$date] ?? 0) + $s['lowres'] + $s['hires'];
          }
      }
  }
  ?>
  <div class="card p-3 mt-3">
    <canvas id="chartStations"></canvas>
  </div>
</div>

<!-- IPS TAB -->
<div class="tab-pane fade" id="tab-ips">
  <div class="mb-2 text-muted" style="font-size:.82rem"><?= t('usage_geoip_note', $lang) ?></div>
  <?php
  // Build country totals for the chart (server-side)
  $country_totals_php = [];
  foreach ($top_ips as $ip => $secs) {
      $country_en = $geoip[$ip]['country'] ?? '';
      $country_disp = ($lang_code === 'nb_NO' && isset($COUNTRY_NO[$country_en])) ? $COUNTRY_NO[$country_en] : $country_en;
      $label   = $country_disp ?: $ip;
      $country_totals_php[$label] = ($country_totals_php[$label] ?? 0) + $secs;
  }
  arsort($country_totals_php);
  $chart_countries = array_slice($country_totals_php, 0, 15, true);
  ?>
  <div class="card p-3 mb-3">
    <canvas id="chartCountries"></canvas>
  </div>
  <div class="card p-3">
    <table class="table table-sm table-hover mb-0" id="ip-table">
      <thead><tr>
        <th><?= t('usage_col_ip', $lang) ?></th><th class="country-cell"><?= t('usage_col_country', $lang) ?></th><th><?= t('usage_col_stream', $lang) ?></th><th><?= t('usage_col_downloaded', $lang) ?></th><th><?= t('usage_col_stations_used', $lang) ?></th>
      </tr></thead>
      <tbody>
      <?php foreach ($top_ips as $ip => $total_s):
          $country_en = $geoip[$ip]['country'] ?? '';
          $country = ($lang_code === 'nb_NO' && isset($COUNTRY_NO[$country_en])) ? $COUNTRY_NO[$country_en] : $country_en;
          $cc      = strtoupper($geoip[$ip]['cc'] ?? '');
          $flag = '';
          if (strlen($cc) === 2 && ctype_alpha($cc)) {
              // Regional indicator symbols start at U+1F1E6 ('A') — encode as 4-byte UTF-8
              $flag = utf8_codepoint(0x1F1E6 + ord($cc[0]) - 65)
                    . utf8_codepoint(0x1F1E6 + ord($cc[1]) - 65);
          }
      ?>
      <tr>
        <td><?= htmlspecialchars($ip) ?></td>
        <td class="country-cell"><?= $flag ? $flag . ' ' : '' ?><?= htmlspecialchars($country ?: ($geoip[$ip] ?? null ? '?' : '…')) ?></td>
        <td><?= fmt_duration($total_s, $lang_code) ?></td>
        <td><?= isset($ip_bytes[$ip]) ? fmt_bytes($ip_bytes[$ip]) : '—' ?></td>
        <td><?php
          $stations_for_ip = [];
          foreach ($stream_by_day as $date => $ips) {
              if (isset($ips[$ip])) {
                  foreach (array_keys($ips[$ip]) as $st) $stations_for_ip[$st] = true;
              }
          }
          $sorted_stations = array_keys($stations_for_ip);
          sort($sorted_stations);
          foreach ($sorted_stations as $st) echo '<span class="badge-station me-1">'.htmlspecialchars($st).'</span>';
        ?></td>
      </tr>
      <?php endforeach; ?>
      </tbody>
    </table>
  </div>
</div>

<!-- ACTIONS TAB -->
<?php if ($access_summary): ?>
<div class="tab-pane fade" id="tab-access">
  <div class="card p-3">
    <h3 style="font-size:1rem" class="mb-3"><?= t('usage_actions_title', $lang) ?></h3>
    <table class="table table-sm table-hover mb-0">
      <thead><tr><th><?= t('usage_col_action', $lang) ?></th><th><?= t('usage_col_count', $lang) ?></th><th style="min-width:140px"></th></tr></thead>
      <tbody>
      <?php
      $max_ac = max(1, max($access_summary));
      foreach ($access_summary as $action => $count):
          $pct = min(100, round($count / $max_ac * 100));
      ?>
      <tr>
        <td><code><?= htmlspecialchars($action) ?></code></td>
        <td><?= number_format($count) ?></td>
        <td><div class="bar-wrap"><div class="bar-fill" style="width:<?= $pct ?>%"></div></div></td>
      </tr>
      <?php endforeach; ?>
      </tbody>
    </table>
  </div>
</div>
<?php endif; ?>

</div><!-- tab-content -->
</div><!-- container -->

<!-- Chart data injected server-side -->
<script>
const dailyDates  = <?= json_encode(array_reverse(array_keys($daily_summary))) ?>;
const dailyStream = <?= json_encode(array_map(fn($r) => round($r['stream_s']/60, 1), array_reverse(array_values($daily_summary)))) ?>;
const dailyBytes  = <?= json_encode(array_map(fn($r) => round($r['bytes']/1048576, 2), array_reverse(array_values($daily_summary)))) ?>;

<?php
// Station colours palette
$stations_list = array_keys($station_totals);
$palette = ['#1f6feb','#3fb950','#f78166','#d2a8ff','#ffa657','#79c0ff','#56d364','#ff7b72','#bc8cff'];
$station_daily_json = [];
foreach ($stations_list as $i => $st) {
    $color = $palette[$i % count($palette)];
    $vals = [];
    foreach (array_reverse(array_keys($daily_summary)) as $d) {
        $vals[] = round(($station_daily[$st][$d] ?? 0) / 60, 1);
    }
    $station_daily_json[] = ['label' => $st, 'data' => $vals, 'color' => $color];
}
?>
const stationDatasets = <?= json_encode($station_daily_json) ?>;
const stationDates    = dailyDates;
</script>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#21262d';

// Daily stream chart
new Chart(document.getElementById('chartStream'), {
  type: 'bar',
  data: {
    labels: dailyDates,
    datasets: [{
      label: <?= json_encode($lang['usage_chart_streaming'] ?? 'Streaming (min)') ?>,
      data: dailyStream,
      backgroundColor: 'rgba(31,111,235,0.7)',
      borderColor: '#1f6feb',
      borderWidth: 1,
      yAxisID: 'y',
    }, {
      label: <?= json_encode($lang['usage_chart_downloaded'] ?? 'Downloaded (MB)') ?>,
      data: dailyBytes,
      type: 'line',
      backgroundColor: 'rgba(63,185,80,0.15)',
      borderColor: '#3fb950',
      borderWidth: 2,
      pointRadius: 3,
      tension: 0.3,
      yAxisID: 'y2',
    }]
  },
  options: {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { labels: { color: '#8b949e' } } },
    scales: {
      x: { ticks: { maxTicksLimit: 20, maxRotation: 45 } },
      y:  { title: { display: true, text: <?= json_encode($lang['usage_chart_minutes'] ?? 'minutes') ?> }, position: 'left' },
      y2: { title: { display: true, text: 'MB' }, position: 'right', grid: { drawOnChartArea: false } },
    }
  }
});

// Station stacked bar chart
new Chart(document.getElementById('chartStations'), {
  type: 'bar',
  data: {
    labels: stationDates,
    datasets: stationDatasets.map(s => ({
      label: s.label,
      data: s.data,
      backgroundColor: s.color + 'b3',
      borderColor: s.color,
      borderWidth: 1,
    }))
  },
  options: {
    responsive: true,
    plugins: { legend: { labels: { color: '#8b949e' } } },
    scales: {
      x: { stacked: true, ticks: { maxTicksLimit: 20, maxRotation: 45 } },
      y: { stacked: true, title: { display: true, text: <?= json_encode($lang['usage_chart_minutes'] ?? 'minutes') ?> } },
    }
  }
});

// Country doughnut chart — data resolved server-side
const palette = ['#1f6feb','#3fb950','#f78166','#d2a8ff','#ffa657','#79c0ff','#56d364','#ff7b72','#bc8cff','#e3b341','#58a6ff','#7ee787','#ffa198','#cae8ff','#ffd700'];
const countryLabels = <?= json_encode(array_keys($chart_countries)) ?>;
const countryValues = <?= json_encode(array_values(array_map(fn($s) => round($s/60), $chart_countries))) ?>;

if (countryLabels.length === 0) {
    document.getElementById('chartCountries').parentElement.innerHTML = '<p class="text-muted p-3"><?= t('usage_no_ip_data', $lang) ?></p>';
} else {
    new Chart(document.getElementById('chartCountries'), {
      type: 'doughnut',
      data: {
        labels: countryLabels,
        datasets: [{ data: countryValues, backgroundColor: palette.slice(0, countryLabels.length) }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'right', labels: { color: '#8b949e' } },
          tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw} min` } }
        }
      }
    });
}
</script>
</body>
</html>
