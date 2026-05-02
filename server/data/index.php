<?php
// --- Configuration ---
$MAX_CONCURRENT_REQUESTS = 8;
$LOCK_DIR = __DIR__ . '/locks';
$PYTHON_SCRIPT = __DIR__ . '/controller.py';
$SATELLITE_SCRIPT = __DIR__ . '/predict_sat.py';
$AIRCRAFT_SCRIPT = __DIR__ . '/predict_flight.py';
$PYTHON_EXECUTABLE = '/usr/bin/python3';
$LANG_DIR = __DIR__ . '/lang';
$DEFAULT_LANG = 'nb_NO';

// --- Setup ---
if (!is_dir($LOCK_DIR)) { mkdir($LOCK_DIR, 0775, true); }

/**
 * Gets the user's real IP address, safely handling requests that come through a proxy.
 * It checks common proxy headers before falling back to the standard REMOTE_ADDR.
 * @return string The user's IP address.
 */
function get_user_ip() {
    if (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {
        return trim(explode(',', $_SERVER['HTTP_X_FORWARDED_FOR'])[0]);
    }
    if (!empty($_SERVER['HTTP_X_REAL_IP'])) {
        return trim($_SERVER['HTTP_X_REAL_IP']);
    }
    return $_SERVER['REMOTE_ADDR'] ?? 'unknown_ip';
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
    $supported_langs = ['nb_NO', 'en_GB', 'de_DE', 'cs_CZ', 'fi_FI'];

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
            die("Language file not found for code: $lang_code");
        }
        $lang_data = file_get_contents($lang_file);
        setcookie('lang', $lang_code, time() + (86400 * 365), "/"); // Set language cookie for 1 year
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' ' . escapeshellarg($action) . ' ' . escapeshellarg($lang_data);
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
        echo shell_exec($command);
        break;

    case 'get_kp_data':
    case 'get_camera_fovs':
    case 'get_lightning_data':
    case 'get_meteor_data':
        header('Content-Type: application/json');
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' ' . escapeshellarg($action);
        echo shell_exec($command);
        break;

    case 'get_station_stats':
        header('Content-Type: application/json');
        $station_id = isset($_GET['station_id']) ? preg_replace('/[^a-zA-Z0-9_]/', '', $_GET['station_id']) : '';
        $start_date = isset($_GET['start_date']) && preg_match('/^\d{4}-\d{2}-\d{2}$/', $_GET['start_date']) ? $_GET['start_date'] : '';
        $end_date = isset($_GET['end_date']) && preg_match('/^\d{4}-\d{2}-\d{2}$/', $_GET['end_date']) ? $_GET['end_date'] : '';
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' ' . escapeshellarg($action) . ' ' . escapeshellarg($station_id);
        if ($start_date !== '') $command .= ' ' . escapeshellarg($start_date);
        if ($start_date !== '' && $end_date !== '') $command .= ' ' . escapeshellarg($end_date);
        echo shell_exec($command);
        break;

    case 'find_passes':
        header('Content-Type: application/json');
        $task_id = uniqid('pass_task_');
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($SATELLITE_SCRIPT) . ' ' . escapeshellarg($task_id) . ' > /dev/null 2>&1 &';
        shell_exec($command);
        echo json_encode(['success' => true, 'task_id' => $task_id]);
        break;

    case 'find_aircraft_crossings':
        header('Content-Type: application/json');
        $task_id = uniqid('aircraft_task_');
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($AIRCRAFT_SCRIPT) . ' ' . escapeshellarg($task_id) . ' > /dev/null 2>&1 &';
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
        
        echo shell_exec($command);
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
        
        echo shell_exec($command);
        break;

    case 'download':
        $lock_files = glob($LOCK_DIR . '/master_task_*.lock');
        if (count($lock_files) >= $MAX_CONCURRENT_REQUESTS) {
            header('HTTP/1.1 503 Service Unavailable');
            die(json_encode(['error' => 'Server is busy, too many concurrent downloads.']));
        }
        $task_id = uniqid('master_task_');
        $post_data = file_get_contents('php://input');
        $user_ip = get_user_ip();

        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' download ' . escapeshellarg($task_id) . ' ' . escapeshellarg($post_data) . ' ' . escapeshellarg($user_ip) . ' > /dev/null 2>&1 &';
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
            echo json_encode(['error' => 'Invalid or missing task ID']);
        }
        break;
}
?>

