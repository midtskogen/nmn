<?php
// --- Configuration ---
// The maximum number of concurrent download requests the server will handle at once.
$MAX_CONCURRENT_REQUESTS = 8;
// Defines the directory for storing lock and status files for background tasks.
$LOCK_DIR = __DIR__ . '/locks';
// Defines the path to the main Python controller script that handles most backend logic.
$PYTHON_SCRIPT = __DIR__ . '/controller.py';
// Defines the path to the Python script for calculating satellite passes.
$SATELLITE_SCRIPT = __DIR__ . '/predict_sat.py';
// Defines the path to the Python script for calculating aircraft crossings.
$AIRCRAFT_SCRIPT = __DIR__ . '/predict_flight.py';
// Defines the path to the Python 3 executable on the server.
$PYTHON_EXECUTABLE = '/usr/bin/python3'; 

// --- Setup ---
// Creates the lock directory if it doesn't already exist.
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

// Determines the requested action from the URL query string, defaulting to 'get_page'.
$action = $_GET['action'] ?? 'get_page';

// --- Router ---
// This switch statement acts as a simple router, directing requests to the appropriate backend logic.
switch ($action) {
    case 'get_page':
    case 'get_stations':
        // For simple actions, directly execute the Python controller and echo its output.
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' ' . escapeshellarg($action);
        echo shell_exec($command);
        break;
    
    // Consolidated cases for actions that fetch and return JSON data.
    case 'get_kp_data':
    case 'get_camera_fovs':
    case 'get_lightning_data':
    case 'get_meteor_data':
        header('Content-Type: application/json');
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' ' . escapeshellarg($action);
        echo shell_exec($command);
        break;
        
    case 'find_passes':
        header('Content-Type: application/json');
        // Generate a unique ID for this asynchronous task.
        $task_id = uniqid('pass_task_');
        // Execute the satellite prediction script as a background process.
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($SATELLITE_SCRIPT) . ' ' . escapeshellarg($task_id) . ' > /dev/null 2>&1 &';
        shell_exec($command);
        // Immediately respond with the task ID so the frontend can start polling.
        echo json_encode(['success' => true, 'task_id' => $task_id]);
        break;

    case 'find_aircraft_crossings':
        header('Content-Type: application/json');
        $task_id = uniqid('aircraft_task_');
        // Execute the aircraft prediction script as a background process.
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($AIRCRAFT_SCRIPT) . ' ' . escapeshellarg($task_id) . ' > /dev/null 2>&1 &';
        shell_exec($command);
        echo json_encode(['success' => true, 'task_id' => $task_id]);
        break;
        
    case 'pass_status':
    case 'stream_status':
    case 'aircraft_status':
        header('Content-Type: application/json');
        $task_id = $_GET['id'] ?? null;
        
        // This logic handles status polling for different types of async tasks.
        // It maps the action to the expected task ID prefix for validation.
        $prefix_map = [
            'pass_status' => 'pass_task_',
            'stream_status' => 'stream_',
            'aircraft_status' => 'aircraft_task_'
        ];
        $prefix = $prefix_map[$action] ?? null;
        
        // Validate the task ID format to prevent directory traversal or other security issues.
        if ($prefix && $task_id && preg_match('/^' . $prefix . '[a-zA-Z0-9_.-]+$/', $task_id)) {
            $status_file = $LOCK_DIR . '/' . $task_id . '.json';
            if (file_exists($status_file)) {
                // If the status file exists, return its content.
                readfile($status_file);
            } else {
                // Otherwise, return a default 'pending' status.
                $default_message = ($action === 'stream_status') ?
                    ['status' => 'pending', 'message' => 'Venter på svar...'] : ['status' => 'pending'];
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

        // Basic validation of input parameters.
        if (!$station_id || !$camera_num || !preg_match('/^ams\d+$/', $station_id) || !ctype_digit($camera_num)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid or missing station_id or camera_num']);
            exit;
        }

        $task_id = uniqid('stream_');
        $user_ip = get_user_ip();
        // Execute the stream relay script as a background process, passing all necessary parameters.
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
        // Validate the task ID before executing the stop command.
        if (!$task_id || !preg_match('/^stream_[a-zA-Z0-9_.-]+$/', $task_id)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid or missing task_id']);
            exit;
        }

        // Execute the stop stream command in the background.
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' stop_stream ' . escapeshellarg($task_id) . ' > /dev/null 2>&1 &';
        shell_exec($command);
        echo json_encode(['success' => true, 'message' => 'Stop signal sent.']);
        break;
        
    case 'fetch_grid':
        header('Content-Type: application/json');
        $stream_task_id = $_GET['stream_task_id'] ?? null;
        $station_id = $_GET['station_id'] ?? null;
        $cam_num = $_GET['cam_num'] ?? null;

        // Validate all parameters before fetching the grid file.
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
        
    case 'download':
        // Enforce a concurrency limit to prevent server overload.
        $lock_files = glob($LOCK_DIR . '/master_task_*.lock');
        if (count($lock_files) >= $MAX_CONCURRENT_REQUESTS) {
            header('HTTP/1.1 503 Service Unavailable');
            die(json_encode(['error' => 'Serveren er opptatt, for mange samtidige nedlastinger.']));
        }
        $task_id = uniqid('master_task_');
        $post_data = file_get_contents('php://input');
        $user_ip = get_user_ip();

        // Start the download coordinator script in the background.
        $command = $PYTHON_EXECUTABLE . ' ' . escapeshellarg($PYTHON_SCRIPT) . ' download ' . escapeshellarg($task_id) . ' ' . escapeshellarg($post_data) . ' ' . escapeshellarg($user_ip) . ' > /dev/null 2>&1 &';
        shell_exec($command);
        echo json_encode(['success' => true, 'task_id' => $task_id]);
        break;

    case 'status':
    case 'cancel':
    case 'cleanup':
        $task_id = $_GET['id'] ?? null;
        // A robust regex to validate the task ID for any of the general task management actions.
        if ($task_id && preg_match('/^(master_task|task|pass_task|stream|aircraft_task)_[a-zA-Z0-9_.-]+$/', $task_id)) {
            header('Content-Type: application/json');
            if ($action === 'status') {
                // This is a generic status endpoint for download tasks.
                $status_file = $LOCK_DIR . '/' . $task_id . '.json';
                if (file_exists($status_file)) {
                    readfile($status_file);
                } else {
                    echo json_encode(['status' => 'pending']);
                }
            } else { // Handles 'cancel' and 'cleanup' actions.
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
