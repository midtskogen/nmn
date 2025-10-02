/**
 * A module to handle all API communication with the backend server.
 * It abstracts away the details of making HTTP requests and polling for asynchronous tasks.
 */

// All API calls are routed through index.php with an 'action' parameter.
const API_BASE = 'index.php?action=';

/**
 * Fetches all initial data required to bootstrap the application on startup.
 * This includes station information, camera fields of view, and recent observational data.
 * It uses Promise.all to fetch all resources in parallel for faster loading.
 * @returns {Promise<object>} A promise that resolves to an object containing all initial data.
 */
export async function fetchInitialData() {
    try {
        const [stations, fovs, meteors, kpData, lightning] = await Promise.all([
            fetch(`${API_BASE}get_stations`).then(r => r.json()),
            fetch(`${API_BASE}get_camera_fovs`).then(r => r.json()),
            fetch(`${API_BASE}get_meteor_data`).then(r => r.json()),
            fetch(`${API_BASE}get_kp_data`).then(r => r.json()),
            fetch(`${API_BASE}get_lightning_data`).then(r => r.json())
        ]);
        return { stations, fovs, meteors, kpData, lightning };
    } catch (error) {
        console.error("Error during initial data load:", error);
        throw new Error("Could not load initial application data.");
    }
}

/**
 * Initiates a long-running, asynchronous task on the backend, such as calculating
 * satellite passes or finding aircraft crossings. It returns a task ID that can be
 * used for polling its status.
 * @param {string} action - The API action to trigger (e.g., 'find_passes').
 * @param {function} pollFn - The function to call for polling the status of the task. This function will be passed the new task_id.
 */
async function startAsyncTask(action, pollFn) {
    const response = await fetch(`${API_BASE}${action}`);
    const data = await response.json();
    if (data.success && data.task_id) {
        // If the task was started successfully, begin polling for its status.
        pollFn(data.task_id);
    } else {
        throw new Error(data.error || `Could not start task: ${action}.`);
    }
}

/**
 * Polls the status of a generic asynchronous task (like downloads or pass finding)
 * at a regular interval. It invokes different callbacks based on the task's status.
 * @param {string} taskId - The ID of the task to poll.
 * @param {string} statusAction - The API action for checking the task's status (e.g., 'pass_status').
 * @param {object} callbacks - An object containing `onProgress`, `onComplete`, and `onError` callback functions.
 * @returns {number} The interval ID for the poller, which can be used with `clearInterval` to stop polling.
 */
function pollTaskStatus(taskId, statusAction, { onProgress, onComplete, onError }) {
    const intervalId = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}${statusAction}&id=${taskId}`);
            const data = await response.json();
            if (!data || !data.status) return; // Ignore empty or invalid responses.

            if (data.status === 'progress') {
                if (onProgress) onProgress(data);
            } else if (data.status === 'complete') {
                clearInterval(intervalId); // Stop polling once the task is complete.
                if (onComplete) onComplete(data);
            } else if (data.status === 'error') {
                clearInterval(intervalId); // Stop polling on error.
                if (onError) onError(data);
            }
        } catch (error) {
            clearInterval(intervalId); // Stop polling on network or parsing errors.
            if (onError) onError({ message: error.message });
        }
    }, 2000); // Poll every 2 seconds.
    return intervalId;
}

// --- Specific Task Functions ---
// These functions provide a cleaner, more specific interface for starting common async tasks.

/**
 * Initiates the process of finding all visible satellite passes.
 * @param {object} callbacks - Callbacks for `onProgress`, `onComplete`, and `onError`.
 */
export function fetchAllPasses(callbacks) {
    const pollFn = (taskId) => pollTaskStatus(taskId, 'pass_status', callbacks);
    return startAsyncTask('find_passes', pollFn);
}

/**
 * Initiates the process of finding all visible aircraft crossings.
 * @param {object} callbacks - Callbacks for `onProgress`, `onComplete`, and `onError`.
 */
export function fetchAllAircraftCrossings(callbacks) {
    const pollFn = (taskId) => pollTaskStatus(taskId, 'aircraft_status', callbacks);
    return startAsyncTask('find_aircraft_crossings', pollFn);
}

/**
 * Starts a file download task on the backend. This can be for a manual time selection
 * or for a specific satellite/aircraft pass.
 * @param {object} payload - The JSON payload containing all download parameters (stations, time, file type, etc.).
 * @returns {Promise<string>} A promise that resolves to the task ID for the download.
 */
export async function startDownload(payload) {
    const response = await fetch(`${API_BASE}download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (data.success && data.task_id) {
        return data.task_id;
    }
    throw new Error(data.error || 'Could not start download.');
}

/**
 * Begins polling the status of a download task.
 * @param {string} taskId - The ID of the download task.
 * @param {object} callbacks - Callbacks for `onProgress`, `onComplete`, and `onError`.
 * @returns {number} The interval ID for the poller.
 */
export function pollDownloadStatus(taskId, callbacks) {
    return pollTaskStatus(taskId, 'status', callbacks);
}

/**
 * Sends a request to the backend to cancel a running task.
 * This is a "fire and forget" request; it doesn't wait for a response.
 * @param {string} taskId - The ID of the task to cancel.
 */
export function cancelTask(taskId) {
    fetch(`${API_BASE}cancel&id=${taskId}`);
}

/**
 * Sends a request to the backend to clean up any temporary files associated with a task.
 * @param {string} taskId - The ID of the task to clean up.
 */
export function cleanupTask(taskId) {
    fetch(`${API_BASE}cleanup&id=${taskId}`);
}


// --- Live Stream Functions ---

/**
 * Initiates a live video stream from a specific camera.
 * @param {string} stationId - The ID of the station.
 * @param {number} cameraNum - The camera number (1-7).
 * @param {string} resolution - The desired resolution ('lowres' or 'hires').
 * @param {boolean} hevcSupported - Whether the user's browser supports HEVC (H.265) video.
 * @returns {Promise<string>} A promise that resolves to the stream task ID.
 */
export async function startStream(stationId, cameraNum, resolution, hevcSupported) {
    const url = `${API_BASE}start_stream&station_id=${stationId}&camera_num=${cameraNum}&resolution=${resolution}&hevc_supported=${hevcSupported}`;
    const response = await fetch(url);
    const data = await response.json();
    if (data.success && data.stream_task_id) {
        return data.stream_task_id;
    }
    throw new Error(data.error || 'Could not start video stream.');
}

/**
 * Fetches the calibration grid image URL for a given stream.
 * @param {string} streamTaskId - The ID of the active stream task.
 * @param {string} stationId - The ID of the station.
 * @param {number} camNum - The camera number.
 * @returns {Promise<object>} A promise that resolves to the JSON response containing the grid URL.
 */
export async function fetchStreamGrid(streamTaskId, stationId, camNum) {
    const url = `${API_BASE}fetch_grid&stream_task_id=${streamTaskId}&station_id=${stationId}&cam_num=${camNum}`;
    const response = await fetch(url);
    return response.json();
}

/**
 * Polls the status of a live stream task, handling its specific lifecycle states
 * like establishing a tunnel, connecting to the camera, and becoming ready.
 * @param {string} taskId - The ID of the stream task to poll.
 * @param {object} callbacks - An object with callbacks for `onStatusUpdate`, `onReady`, and `onError`.
 * @returns {number} The interval ID for the poller.
 */
export function pollStreamStatus(taskId, { onStatusUpdate, onReady, onError }) {
    const intervalId = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}stream_status&id=${taskId}`);
            const data = await response.json();
            if (!data || !data.status) return;

            // Handle the different states of the stream setup process.
            if (data.status === 'ready') {
                clearInterval(intervalId); // Stop polling once the stream is ready.
                if (onReady) onReady(data);
            } else if (data.status === 'camera_failed' || data.status === 'error') {
                clearInterval(intervalId); // Stop polling on failure.
                if (onError) onError(data);
            } else { // Handle intermediate states like 'establishing_tunnel', 'connecting_camera'.
                if (onStatusUpdate) onStatusUpdate(data);
            }
        } catch (error) {
            clearInterval(intervalId);
            if (onError) onError({ message: error.message });
        }
    }, 2000); // Poll every 2 seconds.
    return intervalId;
}

/**
 * Sends a signal to the backend to stop a live stream and clean up its resources.
 * It uses `navigator.sendBeacon` if available, which is more reliable for sending
 * requests when a page is being closed.
 * @param {string} taskId - The ID of the stream task to stop.
 */
export function stopStream(taskId) {
    const payload = new FormData();
    payload.append('task_id', taskId);
    // Use `sendBeacon` for reliability on page unload, as it sends the request
    // asynchronously without expecting a response, making it more likely to succeed.
    if (navigator.sendBeacon) {
        navigator.sendBeacon(`${API_BASE}stop_stream`, payload);
    } else {
        // Fallback to fetch with `keepalive` for older browsers.
        fetch(`${API_BASE}stop_stream`, { method: 'POST', body: payload, keepalive: true });
    }
}
