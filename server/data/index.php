<?php
// --- Configuration ---
// Use the directory of the script as accessed (following symlinks),
// not __DIR__ which resolves to the real path of the source file.
$BASE_DIR = dirname($_SERVER['SCRIPT_FILENAME']);
$MAX_CONCURRENT_REQUESTS = 8;
// Use the web-root data/locks folder so www-data can write status files
// regardless of whether index.php is accessed through /data or nmn/server/data.
$_web_root = isset($_SERVER['DOCUMENT_ROOT']) ? $_SERVER['DOCUMENT_ROOT'] : dirname($BASE_DIR);
$LOCK_DIR = $_web_root . '/data/locks';
$PYTHON_SCRIPT = $BASE_DIR . '/controller.py';
$SATELLITE_SCRIPT = $BASE_DIR . '/predict_sat.py';
$AIRCRAFT_SCRIPT = $BASE_DIR . '/predict_flight.py';
$PYTHON_EXECUTABLE = '/usr/bin/python3';
$LANG_DIR = $BASE_DIR . '/lang';
$DEFAULT_LANG = 'nb_NO';

// --- Setup ---
putenv('NMN_DATA_DIR=' . $BASE_DIR);
putenv('NMN_LOCK_DIR=' . $LOCK_DIR);
// Point Python at the private config/credential directory outside the web root.
$SECRETS_DIR = realpath(__DIR__ . '/../../etc');
if ($SECRETS_DIR && is_dir($SECRETS_DIR)) {
    putenv('NMN_CONFIG_FILE=' . $SECRETS_DIR . '/config.json');
    putenv('NMN_CREDENTIALS_FILE=' . $SECRETS_DIR . '/credentials.json');
}
if (!is_dir($LOCK_DIR)) { mkdir($LOCK_DIR, 0775, true); }

// --- CSRF protection for state-changing endpoints ---
if (session_status() === PHP_SESSION_NONE) { session_start(); }
if (empty($_SESSION['csrf_token'])) {
    $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
}
$csrf_token = $_SESSION['csrf_token'];
// Release the session lock early; the token value is already captured.
session_write_close();

$STATE_ACTIONS = ['download', 'start_stream', 'cancel', 'cleanup', 'stop_stream', 'request_transcode', 'find_passes', 'find_aircraft_crossings'];

/**
 * Gets the user's real IP address, safely handling requests that come through a proxy.
 * It checks common proxy headers before falling back to the standard REMOTE_ADDR.
 * @return string The user's IP address.
 */
function get_user_ip() {
    // Only trust proxy headers when the immediate peer is a trusted proxy;
    // otherwise a client can spoof X-Forwarded-For to bypass per-IP quotas.
    $trusted_proxies = ['127.0.0.1', '::1'];
    $remote = $_SERVER['REMOTE_ADDR'] ?? '';
    if (in_array($remote, $trusted_proxies, true)) {
        if (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {
            $candidate = trim(explode(',', $_SERVER['HTTP_X_FORWARDED_FOR'])[0]);
            if (filter_var($candidate, FILTER_VALIDATE_IP)) {
                return $candidate;
            }
        }
        if (!empty($_SERVER['HTTP_X_REAL_IP'])) {
            $candidate = trim($_SERVER['HTTP_X_REAL_IP']);
            if (filter_var($candidate, FILTER_VALIDATE_IP)) {
                return $candidate;
            }
        }
    }
    return $remote !== '' ? $remote : 'unknown_ip';
}

/**
 * Determines the desired language through a prioritized process:
 * 1. User's language cookie (explicit choice).
 * 2. User's browser 'Accept-Language' header.
 * 3. User's country via IP address (GeoIP lookup).
 * 4. Hardcoded default language.
 * @param string $default_lang The default language code to use as a fallback.
 * @return string The determined and validated language code.
 */
function get_language($default_lang) {
    $supported_langs = ['nb_NO', 'en_GB', 'de_DE', 'cs_CZ', 'fi_FI', 'lv_LV'];

    // Priority 1: Check for an existing language cookie.
    if (isset($_COOKIE['lang']) && in_array($_COOKIE['lang'], $supported_langs)) {
        return $_COOKIE['lang'];
    }

    // Priority 2: Check the browser's Accept-Language header.
    if (isset($_SERVER['HTTP_ACCEPT_LANGUAGE'])) {
        $browser_lang_code = substr($_SERVER['HTTP_ACCEPT_LANGUAGE'], 0, 5);
        $browser_lang_code = str_replace('-', '_', $browser_lang_code);
        if (in_array($browser_lang_code, $supported_langs)) {
            return $browser_lang_code;
        }
        // Fallback for partial codes like 'en'
        $short_code = substr($browser_lang_code, 0, 2);
        foreach ($supported_langs as $supported) {
            if (substr($supported, 0, 2) === $short_code) {
                return $supported;
            }
        }
    }

    // Priority 3: Check the user's country via their IP address.
    $country_to_lang_map = [
        // Norwegian & Scandinavian countries
        'NO' => 'nb_NO', // Norway
        'SE' => 'nb_NO', // Sweden
        'DK' => 'nb_NO', // Denmark
        'FI' => 'fi_FI', // Finland

        // English-speaking countries
        'GB' => 'en_GB', // United Kingdom
        'US' => 'en_GB', // United States
        'CA' => 'en_GB', // Canada
        'AU' => 'en_GB', // Australia
        'NZ' => 'en_GB', // New Zealand
        'IE' => 'en_GB', // Ireland

        // German-speaking countries
        'DE' => 'de_DE', // Germany
        'AT' => 'de_DE', // Austria
        'CH' => 'de_DE', // Switzerland

        // Czech & Slovak
        'CZ' => 'cs_CZ', // Czech Republic
        'SK' => 'cs_CZ', // Slovakia

        // Latvian
        'LV' => 'lv_LV', // Latvia
    ];

    $user_ip = get_user_ip();
    // Use a free GeoIP API to get the country code.
    // Note: In a production environment, you might consider a more robust service or a local database (like MaxMind GeoLite2).
    // The '@' suppresses errors if the API call fails.
    $ctx = stream_context_create([
        'http' => [
            'timeout' => 1,
        ],
        'https' => [
            'timeout' => 1,
        ],
    ]);
    $geo_data_json = @file_get_contents("https://ip-api.com/json/{$user_ip}?fields=countryCode,status", false, $ctx);
    if ($geo_data_json) {
        $geo_data = json_decode($geo_data_json);
        if ($geo_data && $geo_data->status === 'success' && isset($country_to_lang_map[$geo_data->countryCode])) {
            return $country_to_lang_map[$geo_data->countryCode];
        }
    }

    // Priority 4: Return the hardcoded default language.
    return $default_lang;
}

$action = $_GET['action'] ?? 'get_page';

// --- Access logging ---
// Append a compact entry to access_log.json for the usage stats page.
// Skip tile requests (too frequent) and internal polling actions.
$_skip_log = ['tile', 'check_status', 'get_stream_status'];
if (!in_array($action, $_skip_log, true)) {
    $_log_file = $BASE_DIR . '/access_log.json';
    // Rotate the log once it exceeds a reasonable size (50 MB).
    if (file_exists($_log_file) && filesize($_log_file) > 50 * 1024 * 1024) {
        @rename($_log_file, $_log_file . '.old');
    }
    $_log_entry = json_encode([
        'ts'     => date('Y-m-d H:i:s'),
        'date'   => date('Y-m-d'),
        'ip'     => get_user_ip(),
        'action' => $action,
        'station'=> $_GET['station_id'] ?? ($_GET['station'] ?? null),
    ]) . "\n";
    @file_put_contents($_log_file, $_log_entry, FILE_APPEND | LOCK_EX);
}

// --- CSRF gate for state-changing requests ---
if (in_array($action, $STATE_ACTIONS, true)) {
    $supplied_token = $_GET['csrf_token'] ?? $_POST['csrf_token'] ?? $_SERVER['HTTP_X_CSRF_TOKEN'] ?? '';
    if (!is_string($supplied_token) || !hash_equals($csrf_token, $supplied_token)) {
        http_response_code(403);
        echo json_encode(['error' => 'invalid_csrf']);
        exit;
    }
}

// --- Router ---
switch ($action) {
    case 'tile':
        $key = getenv('MAPTILER_KEY') ?: '';
        if ($key === '') {
            http_response_code(500);
            echo 'Missing MAPTILER_KEY';
            break;
        }

        $tile_cache_dir = rtrim(sys_get_temp_dir(), '/') . '/nmn_tile_cache';
        if (!is_dir($tile_cache_dir)) {
            @mkdir($tile_cache_dir, 0775, true);
        }

        $type = $_GET['type'] ?? '';
        $z = $_GET['z'] ?? null;
        $x = $_GET['x'] ?? null;
        $y = $_GET['y'] ?? null;

        if (!in_array($type, ['satellite', 'backdrop', 'hybrid'], true) || !ctype_digit((string)$z) || !ctype_digit((string)$x) || !ctype_digit((string)$y)) {
            http_response_code(400);
            echo 'Invalid tile request';
            break;
        }

        $z = (int)$z;
        $x = (int)$x;
        $y = (int)$y;

        if ($z < 0 || $z > 12 || $x < 0 || $y < 0) {
            http_response_code(400);
            echo 'Invalid tile coordinates';
            break;
        }

        if ($type === 'satellite') {
            $upstream = "https://api.maptiler.com/maps/satellite/{$z}/{$x}/{$y}.jpg?key=" . rawurlencode($key);
            $contentType = 'image/jpeg';
        } elseif ($type === 'hybrid') {
            $upstream = "https://api.maptiler.com/maps/hybrid/{$z}/{$x}/{$y}.png?key=" . rawurlencode($key);
            $contentType = 'image/png';
        } else {
            $upstream = "https://api.maptiler.com/maps/backdrop/{$z}/{$x}/{$y}@2x.png?key=" . rawurlencode($key);
            $contentType = 'image/png';
        }

        $cache_ttl = 86400;
        $cache_key = hash('sha256', $type . '|' . $z . '|' . $x . '|' . $y);
        $cache_file = $tile_cache_dir . '/' . $cache_key;
        $etag = '"' . $cache_key . '"';
        header('ETag: ' . $etag);

        if (isset($_SERVER['HTTP_IF_NONE_MATCH']) && trim($_SERVER['HTTP_IF_NONE_MATCH']) === $etag && file_exists($cache_file)) {
            http_response_code(304);
            break;
        }

        if (file_exists($cache_file) && (time() - filemtime($cache_file)) < $cache_ttl) {
            header('Content-Type: ' . $contentType);
            header('Cache-Control: public, max-age=86400');
            readfile($cache_file);
            break;
        }

        $ctx = stream_context_create([
            'http' => [
                'timeout' => 5,
                'header' => "User-Agent: norskmeteornettverk.no\r\nReferer: https://norskmeteornettverk.no/\r\n",
            ],
            'https' => [
                'timeout' => 5,
                'header' => "User-Agent: norskmeteornettverk.no\r\nReferer: https://norskmeteornettverk.no/\r\n",
            ],
        ]);

        $data = false;
        $httpCode = null;
        $curlErr = null;
        if (function_exists('curl_init')) {
            $ch = curl_init($upstream);
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
            curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);
            curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 5);
            curl_setopt($ch, CURLOPT_TIMEOUT, 10);
            curl_setopt($ch, CURLOPT_USERAGENT, 'norskmeteornettverk.no');
            curl_setopt($ch, CURLOPT_HTTPHEADER, ['Referer: https://norskmeteornettverk.no/']);
            $data = curl_exec($ch);
            $curlErr = curl_error($ch);
            $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
            curl_close($ch);

            if ($data === false || $httpCode < 200 || $httpCode >= 300) {
                $data = false;
            }
        }

        if ($data === false) {
            $data = @file_get_contents($upstream, false, $ctx);
        }
        if ($data === false) {
            http_response_code(502);
            echo 'Tile fetch failed';
            break;
        }

        header('Content-Type: ' . $contentType);
        header('Cache-Control: public, max-age=86400');

        if (is_dir($tile_cache_dir) && is_writable($tile_cache_dir)) {
            $tmp = $cache_file . '.' . uniqid('tmp_', true);
            @file_put_contents($tmp, $data, LOCK_EX);
            @rename($tmp, $cache_file);
        }
        echo $data;
        break;

    case 'get_page':
        $lang_code = get_language($DEFAULT_LANG);
        $lang_file = $LANG_DIR . '/' . $lang_code . '.json';
        if (!file_exists($lang_file)) {
            http_response_code(500);
            die("Language file not found for code: " . htmlspecialchars($lang_code, ENT_QUOTES, 'UTF-8'));
        }
        setcookie('lang', $lang_code, [
            'expires'  => time() + 86400 * 365,
            'path'     => '/',
            'httponly' => true,
            'secure'   => true,
            'samesite' => 'Lax',
        ]);
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' ' . escapeshellarg($action) . ' ' . escapeshellarg($lang_file) . ' ' . escapeshellarg($csrf_token);
        echo shell_exec($command);
        break;

    case 'get_lang':
        $lang_code = get_language($DEFAULT_LANG);
        $lang_file = $LANG_DIR . '/' . $lang_code . '.json';
        if (file_exists($lang_file)) {
            header('Content-Type: application/json');
            readfile($lang_file);
        } else {
            http_response_code(404);
            echo json_encode(['error' => 'Language file not found.']);
        }
        break;

    case 'get_stations':
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' ' . escapeshellarg($action);
        $_out = shell_exec($command);
        echo $_out ?: json_encode(['error' => 'Backend returned no output']);
        break;

    case 'get_kp_data':
    case 'get_camera_fovs':
    case 'get_lightning_data':
    case 'get_meteor_data':
        header('Content-Type: application/json');
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' ' . escapeshellarg($action);
        $_out = shell_exec($command);
        echo $_out ?: json_encode(['error' => 'Backend returned no output']);
        break;

    case 'get_station_stats':
        header('Content-Type: application/json');
        $station_id = isset($_GET['station_id']) ? preg_replace('/[^a-zA-Z0-9_]/', '', $_GET['station_id']) : '';
        $start_date = isset($_GET['start_date']) && preg_match('/^\d{4}-\d{2}-\d{2}$/', $_GET['start_date']) ? $_GET['start_date'] : '';
        $end_date = isset($_GET['end_date']) && preg_match('/^\d{4}-\d{2}-\d{2}$/', $_GET['end_date']) ? $_GET['end_date'] : '';
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' ' . escapeshellarg($action) . ' ' . escapeshellarg($station_id);
        if ($start_date !== '') $command .= ' ' . escapeshellarg($start_date);
        if ($start_date !== '' && $end_date !== '') $command .= ' ' . escapeshellarg($end_date);
        $_out = shell_exec($command);
        echo $_out ?: json_encode(['error' => 'Backend returned no output']);
        break;

    case 'find_passes':
        header('Content-Type: application/json');
        $task_id = uniqid('pass_task_');
        // Optional filters: restrict to specific station(s) and/or a shorter
        // time window than the full search range. Both are validated strictly
        // since they're passed through to a shell command.
        $stations_param = isset($_GET['station']) ? $_GET['station'] : '';
        $station_ids = array_filter(array_map('trim', explode(',', $stations_param)), function($s) {
            return $s !== '' && preg_match('/^[a-zA-Z0-9_]+$/', $s);
        });
        $days_param = isset($_GET['days']) && preg_match('/^\d+$/', $_GET['days']) ? (int)$_GET['days'] : null;
        // Explicit ISO8601 UTC range (e.g. from the satellite panel's drag
        // slider), takes precedence over 'days' when present.
        $iso_pattern = '/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$/';
        $start_param = isset($_GET['start']) && preg_match($iso_pattern, $_GET['start']) ? $_GET['start'] : null;
        $end_param = isset($_GET['end']) && preg_match($iso_pattern, $_GET['end']) ? $_GET['end'] : null;

        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($SATELLITE_SCRIPT) . ' ' . escapeshellarg($task_id);
        if (!empty($station_ids)) $command .= ' --station ' . escapeshellarg(implode(',', $station_ids));
        if ($start_param !== null) $command .= ' --start ' . escapeshellarg($start_param);
        if ($end_param !== null) $command .= ' --end ' . escapeshellarg($end_param);
        if ($start_param === null && $end_param === null && $days_param !== null && $days_param > 0) $command .= ' --days ' . escapeshellarg($days_param);
        $command .= ' > /dev/null 2>&1 &';
        shell_exec($command);
        echo json_encode(['success' => true, 'task_id' => $task_id]);
        break;

    case 'find_aircraft_crossings':
        header('Content-Type: application/json');
        $task_id = uniqid('aircraft_task_');
        // Optional filters: restrict to specific station(s) and/or a shorter
        // time window than the full search range. Both are validated strictly
        // since they're passed through to a shell command.
        $stations_param = isset($_GET['station']) ? $_GET['station'] : '';
        $station_ids = array_filter(array_map('trim', explode(',', $stations_param)), function($s) {
            return $s !== '' && preg_match('/^[a-zA-Z0-9_]+$/', $s);
        });
        $days_param = isset($_GET['days']) && preg_match('/^\d+$/', $_GET['days']) ? (int)$_GET['days'] : null;
        // Explicit ISO8601 UTC range (e.g. from the aircraft panel's drag slider).
        $iso_pattern = '/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$/';
        $start_param = isset($_GET['start']) && preg_match($iso_pattern, $_GET['start']) ? $_GET['start'] : null;
        $end_param = isset($_GET['end']) && preg_match($iso_pattern, $_GET['end']) ? $_GET['end'] : null;

        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($AIRCRAFT_SCRIPT) . ' ' . escapeshellarg($task_id);
        if (!empty($station_ids)) $command .= ' --station ' . escapeshellarg(implode(',', $station_ids));
        if ($start_param !== null) $command .= ' --start ' . escapeshellarg($start_param);
        if ($end_param !== null) $command .= ' --end ' . escapeshellarg($end_param);
        if ($start_param === null && $end_param === null && $days_param !== null && $days_param > 0) $command .= ' --days ' . escapeshellarg($days_param);
        $command .= ' > /dev/null 2>&1 &';
        shell_exec($command);
        echo json_encode(['success' => true, 'task_id' => $task_id]);
        break;

    case 'pass_status':
    case 'stream_status':
    case 'aircraft_status':
        header('Content-Type: application/json');
        $task_id = $_GET['id'] ?? null;
        
        $prefix_map = [
            'pass_status' => 'pass_task_',
            'stream_status' => 'stream_',
            'aircraft_status' => 'aircraft_task_'
        ];
        $prefix = $prefix_map[$action] ?? null;
        
        if ($prefix && $task_id && preg_match('/^' . $prefix . '[a-zA-Z0-9_.-]+$/', $task_id)) {
            $status_file = $LOCK_DIR . '/' . $task_id . '.json';
            if (file_exists($status_file)) {
                readfile($status_file);
            } else {
                $default_message = ($action === 'stream_status') ?
                    ['status' => 'pending', 'message' => 'Waiting for response...'] : ['status' => 'pending'];
                echo json_encode($default_message);
            }
        } else {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid or missing task ID']);
        }
        break;

    case 'start_stream':
        header('Content-Type: application/json');
        $station_id = $_GET['station_id'] ?? null;
        $camera_num = $_GET['camera_num'] ?? null;
        $resolution = $_GET['resolution'] ?? 'lowres';
        $hevc_supported = $_GET['hevc_supported'] ?? 'false';

        if (!$station_id || !$camera_num || !preg_match('/^ams\d+$/', $station_id) || !ctype_digit($camera_num)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid or missing station_id or camera_num']);
            exit;
        }

        $task_id = uniqid('stream_');
        $user_ip = get_user_ip();
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' _internal_start_stream '
            . escapeshellarg($task_id) . ' '
            . escapeshellarg($station_id) . ' '
            . escapeshellarg($camera_num) . ' '
            . escapeshellarg($resolution) . ' '
            . escapeshellarg($hevc_supported) . ' '
            . escapeshellarg($user_ip) . ' > /dev/null 2>&1 &';

        shell_exec($command);
        echo json_encode(['success' => true, 'stream_task_id' => $task_id]);
        break;

    case 'stop_stream':
        header('Content-Type: application/json');
        $task_id = $_POST['task_id'] ?? null;
        if (!$task_id || !preg_match('/^stream_[a-zA-Z0-9_.-]+$/', $task_id)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid or missing task_id']);
            exit;
        }

        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' stop_stream ' . escapeshellarg($task_id) . ' > /dev/null 2>&1 &';
        shell_exec($command);
        echo json_encode(['success' => true, 'message' => 'Stop signal sent.']);
        break;

    case 'request_transcode':
        header('Content-Type: application/json');
        $task_id = $_GET['task_id'] ?? null;
        if (!$task_id || !preg_match('/^stream_[a-zA-Z0-9_.-]+$/', $task_id)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid or missing task_id']);
            exit;
        }

        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' request_transcode ' . escapeshellarg($task_id);
        $_out = shell_exec($command);
        echo $_out ?: json_encode(['error' => 'Backend returned no output']);
        break;

    case 'fetch_grid':
        header('Content-Type: application/json');
        $stream_task_id = $_GET['stream_task_id'] ?? null;
        $station_id = $_GET['station_id'] ?? null;
        $cam_num = $_GET['cam_num'] ?? null;

        if (!$stream_task_id || !$station_id || !$cam_num || !preg_match('/^stream_[a-zA-Z0-9_.-]+$/', $stream_task_id) || !preg_match('/^ams\d+$/', $station_id) || !ctype_digit($cam_num)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid or missing parameters']);
            exit;
        }

        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' fetch_grid '
            . escapeshellarg($stream_task_id) . ' '
            . escapeshellarg($station_id) . ' '
            . escapeshellarg($cam_num);
        
        $_out = shell_exec($command);
        echo $_out ?: json_encode(['error' => 'Backend returned no output']);
        break;

    case 'fetch_annotation':
        header('Content-Type: application/json');
        $stream_task_id = $_GET['stream_task_id'] ?? null;
        $station_id = $_GET['station_id'] ?? null;
        $cam_num = $_GET['cam_num'] ?? null;

        if (!$stream_task_id || !$station_id || !$cam_num || !preg_match('/^stream_[a-zA-Z0-9_.-]+$/', $stream_task_id) || !preg_match('/^ams\d+$/', $station_id) || !ctype_digit($cam_num)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid or missing parameters']);
            exit;
        }

        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' fetch_annotation '
            . escapeshellarg($stream_task_id) . ' '
            . escapeshellarg($station_id) . ' '
            . escapeshellarg($cam_num);
        
        $_out = shell_exec($command);
        echo $_out ?: json_encode(['error' => 'Backend returned no output']);
        break;

    case 'fetch_stitch_grid':
        header('Content-Type: application/json');
        $projection = $_GET['projection'] ?? null;  // 'eq' or 'fe'
        $resolution  = $_GET['resolution']  ?? null;  // 'hires' or 'lowres'
        if (!in_array($projection, ['eq', 'fe'], true) || !in_array($resolution, ['hires', 'lowres'], true)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid parameters']);
            exit;
        }
        $grid_file = "grid_{$projection}_" . ($resolution === 'lowres' ? 'sd' : 'hd') . ".png";
        $grid_path = $BASE_DIR . '/' . $grid_file;
        if (file_exists($grid_path)) {
            echo json_encode(['success' => true, 'grid_url' => $grid_file]);
        } else {
            echo json_encode(['success' => false, 'error' => 'grid_not_found']);
        }
        break;

    case 'fetch_stitch_cam_boundaries':
        header('Content-Type: application/json');
        $station_id = $_GET['station_id'] ?? null;
        $projection  = $_GET['projection']  ?? null;  // 'eq' or 'fe'
        $resolution  = $_GET['resolution']  ?? 'hires';  // 'hires' or 'lowres'
        if (!in_array($resolution, ['hires', 'lowres'], true)) $resolution = 'hires';
        if (!$station_id || !in_array($projection, ['eq', 'fe'], true) || !preg_match('/^[a-zA-Z0-9_-]+$/', $station_id)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid parameters']);
            exit;
        }
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT)
            . ' fetch_stitch_cam_boundaries '
            . escapeshellarg($station_id) . ' '
            . escapeshellarg($projection) . ' '
            . escapeshellarg($resolution);
        $_out = shell_exec($command);
        echo $_out ?: json_encode(['error' => 'Backend returned no output']);
        break;

    case 'fetch_archive_grid':
        header('Content-Type: application/json');
        $station_id = $_GET['station_id'] ?? null;
        $camera_num = $_GET['camera_num'] ?? null;
        $timestamp = $_GET['timestamp'] ?? null;

        if (!$station_id || !$camera_num || !$timestamp || !preg_match('/^[A-Z]{3}$/', $station_id) || !ctype_digit($camera_num) || !preg_match('/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/', $timestamp)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid or missing parameters']);
            exit;
        }

        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' fetch_archive_grid '
            . escapeshellarg($station_id) . ' '
            . escapeshellarg($camera_num) . ' '
            . escapeshellarg($timestamp);

        $_out = shell_exec($command);
        echo $_out ?: json_encode(['error' => 'Backend returned no output']);
        break;

    case 'fetch_archive_annotation':
        header('Content-Type: application/json');
        $station_id = $_GET['station_id'] ?? null;
        $camera_num = $_GET['camera_num'] ?? null;
        $timestamp = $_GET['timestamp'] ?? null;

        if (!$station_id || !$camera_num || !$timestamp || !preg_match('/^[A-Z]{3}$/', $station_id) || !ctype_digit($camera_num) || !preg_match('/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/', $timestamp)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid or missing parameters']);
            exit;
        }

        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' fetch_archive_annotation '
            . escapeshellarg($station_id) . ' '
            . escapeshellarg($camera_num) . ' '
            . escapeshellarg($timestamp);

        $_out = shell_exec($command);
        echo $_out ?: json_encode(['error' => 'Backend returned no output']);
        break;

    case 'fetch_archive_mask':
        header('Content-Type: application/json');
        $station_id = $_GET['station_id'] ?? null;
        $camera_num = $_GET['camera_num'] ?? null;

        if (!$station_id || !$camera_num || !preg_match('/^[A-Z]{3}$/', $station_id) || !ctype_digit($camera_num)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid or missing parameters']);
            exit;
        }

        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' fetch_archive_mask '
            . escapeshellarg($station_id) . ' '
            . escapeshellarg($camera_num);

        $_out = shell_exec($command);
        echo $_out ?: json_encode(['error' => 'Backend returned no output']);
        break;

    case 'enhance_filter':
        header('Content-Type: application/json');
        $image = $_GET['image'] ?? null;
        $threshold = $_GET['threshold'] ?? '0';

        // Only simple filenames inside the download directory are allowed.
        if (!$image || !preg_match('#^download/[A-Za-z0-9_.-]+\.(jpg|jpeg|png)$#i', $image) || !ctype_digit($threshold)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid or missing parameters']);
            exit;
        }

        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' enhance_filter '
            . escapeshellarg($image) . ' '
            . escapeshellarg($threshold);

        $_out = shell_exec($command);
        echo $_out ?: json_encode(['error' => 'Backend returned no output']);
        break;

    case 'download':
        // Enforce a sane payload size before writing it to disk.
        $max_payload_bytes = 5 * 1024 * 1024;
        $raw_post = file_get_contents('php://input');
        if (strlen($raw_post) > $max_payload_bytes) {
            http_response_code(413);
            echo json_encode(['error' => 'Download payload too large']);
            exit;
        }

        // Use a counting semaphore so the limit is enforced atomically.
        $sem_file = $LOCK_DIR . '/download_semaphore.lock';
        $sem = fopen($sem_file, 'c');
        if (!$sem || !flock($sem, LOCK_EX)) {
            header('HTTP/1.1 503 Service Unavailable');
            die(json_encode(['error' => 'Server is busy, could not acquire lock.']));
        }
        $lock_files = glob($LOCK_DIR . '/master_task_*.lock');
        if (count($lock_files) >= $MAX_CONCURRENT_REQUESTS) {
            flock($sem, LOCK_UN);
            fclose($sem);
            header('HTTP/1.1 503 Service Unavailable');
            die(json_encode(['error' => 'Server is busy, too many concurrent downloads.']));
        }
        $task_id = uniqid('master_task_');
        $payload_file = tempnam($LOCK_DIR, 'payload_');
        file_put_contents($payload_file, $raw_post, LOCK_EX);
        // Create the lock marker before releasing the semaphore.
        touch($LOCK_DIR . '/' . $task_id . '.lock');
        flock($sem, LOCK_UN);
        fclose($sem);

        $user_ip = get_user_ip();
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' download ' . escapeshellarg($task_id) . ' ' . escapeshellarg($payload_file) . ' ' . escapeshellarg($user_ip) . ' > /dev/null 2>&1 &';
        shell_exec($command);
        echo json_encode(['success' => true, 'task_id' => $task_id]);
        break;

    case 'status':
    case 'cancel':
    case 'cleanup':
        $task_id = $_GET['id'] ?? null;
        if ($task_id && preg_match('/^(master_task|task|pass_task|stream|aircraft_task)_[a-zA-Z0-9_.-]+$/', $task_id)) {
            header('Content-Type: application/json');
            if ($action === 'status') {
                $status_file = $LOCK_DIR . '/' . $task_id . '.json';
                if (file_exists($status_file)) {
                    readfile($status_file);
                } else {
                    echo json_encode(['status' => 'pending']);
                }
            } else {
                $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' ' . escapeshellarg($action) . ' ' . escapeshellarg($task_id);
                shell_exec($command);
                echo json_encode(['success' => true]);
            }
        } else {
            http_response_code(400);
            header('Content-Type: application/json');
            echo json_encode(['error' => 'Invalid or missing task ID']);
        }
        break;
}
?>

