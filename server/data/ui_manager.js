import { createEl, isHevcSupported } from './utils.js';
import { getSunTimes, getSunAltitude, calculateBearing, destinationPoint } from './calculations.js';
import * as api from './api.js';
import { airlineCodes } from './airline_codes.js';
// --- Module-scoped variables ---
let dom = {}; // A cache for frequently accessed DOM elements.
let t = (key) => key; // The translation function, initialized to a fallback.
let hls = null;
// Holds the HLS.js instance for playing HLS video streams.
let streamCountdownInterval = null;
// Interval ID for the stream timeout countdown.
let bitrateUpdateInterval = null;
// Interval ID for updating bitrate in the status line.
let activeStreamTaskId = null;
// The task ID of the currently active stream.
let stopStreamTimeout = null;
// Timeout ID to automatically close the modal.
let streamStatusPoller = null; // Interval ID for polling the stream's status.
let onFullscreenChange = null; // Holds the fullscreen change event handler.
let lastModalDimensions = null; // Stores dimensions for smooth prev/next navigation.
let currentMediaList = []; // Global list of all media items for navigation - updated dynamically.
let previewStationsData = null; // Station data for preview modals

let meteorReportExistenceCache = new Map();
let meteorListRenderToken = 0;

let urlCheckInFlight = 0;
const urlCheckQueue = [];
const URL_CHECK_CONCURRENCY = 4;

// History API for back button handling
let modalHistoryState = null;

// Handle browser back button to close modals
window.addEventListener('popstate', (event) => {
    const modalBackdrop = document.getElementById('video-modal-backdrop');
    if (modalBackdrop) {
        // A modal is open, close it instead of navigating
        modalBackdrop.remove();
        // Don't push a new state, just let the history revert
        event.preventDefault();
        event.stopPropagation();
    }
});

function processUrlCheckQueue() {
    while (urlCheckInFlight < URL_CHECK_CONCURRENCY && urlCheckQueue.length > 0) {
        const { run } = urlCheckQueue.shift();
        urlCheckInFlight++;
        run().finally(() => { urlCheckInFlight--; processUrlCheckQueue(); });
    }
}

/**
 * Adds a top drag bar and draggable corner resize handles to a modal content element.
 * @param {HTMLElement} modalContent
 * @param {Function} [onResize] - Optional callback invoked while resizing (rAF-throttled).
 */
function makeModalResizable(modalContent, onResize) {
    // Top drag bar to move the modal
    const dragBar = createEl('div', { className: 'modal-drag-bar' });
    modalContent.prepend(dragBar);

    const corners = [
        { cls: 'nw', dx: -1, dy: -1 },
        { cls: 'ne', dx: 1, dy: -1 },
        { cls: 'sw', dx: -1, dy: 1 },
        { cls: 'se', dx: 1, dy: 1 }
    ];
    corners.forEach(({ cls }) => {
        const handle = createEl('div', { className: `modal-resize-handle ${cls}` });
        modalContent.appendChild(handle);
        handle.addEventListener('mousedown', (e) => startResize(e, cls));
    });

    modalContent.style.position = 'relative';

    let isResizing = false;
    let isDragging = false;
    let activeDx = 1;
    let activeDy = 1;
    let startX, startY, startWidth, startHeight, startLeft, startTop;
    let dragOffsetX = 0, dragOffsetY = 0;
    let resizeRafId = null;

    const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

    function startDrag(e) {
        if (e.button !== 0 || document.fullscreenElement) return;
        e.preventDefault();
        e.stopPropagation();
        isDragging = true;
        const rect = modalContent.getBoundingClientRect();
        modalContent.style.position = 'absolute';
        modalContent.style.left = rect.left + 'px';
        modalContent.style.top = rect.top + 'px';
        dragOffsetX = e.clientX - rect.left;
        dragOffsetY = e.clientY - rect.top;
        window.addEventListener('mousemove', onDragMove);
        window.addEventListener('mouseup', onDragUp);
    }

    function onDragMove(e) {
        if (!isDragging) return;
        const newLeft = clamp(e.clientX - dragOffsetX, 0, window.innerWidth - modalContent.offsetWidth);
        const newTop = clamp(e.clientY - dragOffsetY, 0, window.innerHeight - modalContent.offsetHeight);
        modalContent.style.left = newLeft + 'px';
        modalContent.style.top = newTop + 'px';
    }

    function onDragUp() {
        isDragging = false;
        window.removeEventListener('mousemove', onDragMove);
        window.removeEventListener('mouseup', onDragUp);
    }

    dragBar.addEventListener('mousedown', startDrag);

    function startResize(e, cls) {
        e.preventDefault();
        e.stopPropagation();
        if (document.fullscreenElement) return;
        isResizing = true;
        const corner = corners.find(c => c.cls === cls);
        activeDx = corner.dx;
        activeDy = corner.dy;
        startX = e.clientX;
        startY = e.clientY;
        const rect = modalContent.getBoundingClientRect();
        startWidth = rect.width;
        startHeight = rect.height;
        startLeft = rect.left;
        startTop = rect.top;
        modalContent.style.position = 'absolute';
        modalContent.style.left = startLeft + 'px';
        modalContent.style.top = startTop + 'px';
        modalContent.style.width = startWidth + 'px';
        modalContent.style.height = startHeight + 'px';
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
    }

    function onMouseMove(e) {
        if (!isResizing) return;
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        const newWidth = clamp(startWidth + dx * activeDx, 400, window.innerWidth * 0.95);
        const newHeight = clamp(startHeight + dy * activeDy, 300, window.innerHeight * 0.95);

        let offsetLeft = 0;
        let offsetTop = 0;
        if (activeDx < 0) offsetLeft = startWidth - newWidth;
        if (activeDy < 0) offsetTop = startHeight - newHeight;

        modalContent.style.left = (startLeft + offsetLeft) + 'px';
        modalContent.style.top = (startTop + offsetTop) + 'px';
        modalContent.style.width = newWidth + 'px';
        modalContent.style.height = newHeight + 'px';

        if (onResize) {
            if (resizeRafId) cancelAnimationFrame(resizeRafId);
            resizeRafId = requestAnimationFrame(() => {
                resizeRafId = null;
                onResize();
            });
        }
    }

    function onMouseUp() {
        isResizing = false;
        if (resizeRafId) {
            cancelAnimationFrame(resizeRafId);
            resizeRafId = null;
        }
        if (onResize) onResize();
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
    }
}

/**
 * Parses a filename and builds an enhanced title with station info, coordinates, elevation, sun altitude, and ISO timestamp.
 * Filename formats:
 * - Regular: stationCode_camN_YYYYMMDD_HHMM_type.ext
 * - Stitched: stationCode_YYYYMMDD_HHMM_resolution_projection.ext (no camN)
 * @param {string} filename - The filename to parse
 * @returns {string} The enhanced title
 */
function buildEnhancedPreviewTitle(filename, url = null) {
    if (!previewStationsData) return filename;

    // If filename is a short display name (like eqh, fel, eqll, fell, etc.), extract actual filename from URL
    if (url && /^(eq|fe)[hl]{0,2}$/.test(filename)) {
        const urlParts = url.split('/');
        const actualFilename = urlParts[urlParts.length - 1];
        if (actualFilename && actualFilename !== filename) {
            filename = actualFilename;
        }
    }

    // Remove extension first
    const nameWithoutExt = filename.replace(/\.[^.]+$/, '');

    // Parse filename: stationCode_[camN_]YYYYMMDD_HHMM_[resolution_]projection
    const parts = nameWithoutExt.split('_');
    if (parts.length < 3) return filename;

    const stationCode = parts[0];

    // Find date and time parts - they should be 8-digit and 4-digit respectively
    let dateStr = null;
    let timeStr = null;

    for (let i = 1; i < parts.length; i++) {
        const part = parts[i];
        // Check for YYYYMMDD format (8 digits)
        if (/^\d{8}$/.test(part)) {
            dateStr = part;
        }
        // Check for HHMM format (4 digits)
        else if (/^\d{4}$/.test(part)) {
            timeStr = part;
        }
    }

    // Validate date and time strings
    if (!dateStr || !timeStr) {
        return filename;
    }

    // Build ISO timestamp
    const isoTimestamp = `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6, 8)}T${timeStr.slice(0, 2)}:${timeStr.slice(2, 4)}:00Z`;

    // Find station data
    const stationId = Object.keys(previewStationsData).find(id => previewStationsData[id].station?.code === stationCode);
    if (!stationId) return `${filename} | ${isoTimestamp}`;

    const stationInfo = previewStationsData[stationId].station;
    const astronomy = previewStationsData[stationId].astronomy;
    const displayName = stationInfo?.display_name || (stationInfo?.name ? stationInfo.name.charAt(0).toUpperCase() + stationInfo.name.slice(1) : stationCode);

    // Calculate sun altitude at the timestamp
    let sunAltText = '';
    if (astronomy && astronomy.latitude && astronomy.longitude) {
        const timestampDate = new Date(isoTimestamp);
        const sunAlt = getSunAltitude(timestampDate, astronomy.latitude, astronomy.longitude);
        sunAltText = ` | ${t('sun_altitude')}: ${sunAlt.toFixed(1)}°`;
    }

    // Build title with coordinates, elevation, sun altitude, and ISO timestamp
    let titleText = `${displayName}`;
    if (astronomy) {
        const lat = `${astronomy.latitude.toFixed(3)}N`;
        const lon = `${astronomy.longitude.toFixed(3)}E`;
        const elev = astronomy.elevation ? `${astronomy.elevation}m` : '';
        titleText += ` (${lat}, ${lon}${elev ? `, ${elev}` : ''}${sunAltText})`;
    }
    titleText += ` | ${isoTimestamp}`;

    return titleText;
}

async function checkUrlExists(url) {
    if (meteorReportExistenceCache.has(url)) return meteorReportExistenceCache.get(url);

    let resolve;
    const promise = new Promise(r => { resolve = r; });
    meteorReportExistenceCache.set(url, promise);

    urlCheckQueue.push({ run: async () => {
        try {
            let res = await fetch(url, { method: 'HEAD', cache: 'no-store' });
            if (res.status === 405 || res.status === 501) {
                res = await fetch(url, { method: 'GET', cache: 'no-store' });
            }
            resolve(res.ok);
        } catch (e) {
            resolve(false);
        }
    }});
    processUrlCheckQueue();
    return promise;
}
// --- Private Helper Functions ---

/**
 * Populates the time-related select dropdowns in the form with their respective options.
 * This is run once during initialization.
 */
function initFormControls() {
    ['hour', 'minute', 'length', 'interval'].forEach(id => {
        const select = dom[`${id}Select`];
        if (!select) return;
        select.innerHTML = ''; // Clear any existing options.
        const limit = (id === 'hour') ? 24 : 60;
        const start = (id === 'length' || id === 'interval') ? 1 : 0;
        

        for (let i = start; i < (start === 1 ? limit + 1 : limit); i++) {
            select.add(new Option(String(i).padStart(2, '0'), i));
        }
    });
}

// --- Public API ---

/**
 * Initializes the UI Manager.
 * @param {object} domCache - An object containing references to key DOM elements from main.js.
 * @param {function} onResetClick - The callback function to execute when the reset button is clicked.
 * @param {function} translationFunc - The translation function from main.js.
 */
export function initUIManager(domCache, onResetClick, translationFunc) {
    dom = domCache;
    t = translationFunc;
    initFormControls();
    setDefaultFormValues();
    const controlPanelHeader = document.querySelector('#control-panel h2');
    const headerWrapper = createEl('div', { className: 'panel-header-wrapper' });
    const formResetButton = createEl('button', {
        type: 'button',
        id: 'form-reset-button',
        textContent: t('reset_button'),
        onclick: onResetClick
    });
    controlPanelHeader.parentNode.insertBefore(headerWrapper, controlPanelHeader);
    headerWrapper.append(controlPanelHeader, formResetButton);
}

/**
 * Sets the default values for the download form controls on application load.
 */
export function setDefaultFormValues() {
    const today = new Date();
    today.setMinutes(today.getMinutes() - today.getTimezoneOffset());
    const todayISO = today.toISOString().slice(0, 10);
    dom.dateInput.value = todayISO;
    dom.dateInput.max = todayISO;
    dom.dateDisplayInput.value = todayISO;
    dom.hourSelect.value = 0;
    dom.minuteSelect.value = 0;
    dom.lengthSelect.value = 1;
    dom.intervalSelect.value = 1;
    if (dom.durationSelect) dom.durationSelect.value = 1;
    document.querySelectorAll('input[name="cameras"]').forEach(cb => { cb.checked = true; cb.disabled = false; });
    const imageRadio = document.querySelector('input[name="primary_file_type"][value="image"]');
    imageRadio.checked = true;
    imageRadio.dispatchEvent(new Event('change'));
    document.getElementById('high-resolution-switch').checked = false;
    document.getElementById('long-integration-switch').checked = false;
    document.getElementById('long-integration-label').style.display = 'flex';
}

/**
 * Manages the visual state of the main download form and its controls.
 * @param {string} state - The state to set: 'ready', 'downloading', or 'cooldown'.
 */
export function setUIState(state) {
    if (state === 'ready') {
        dom.downloadButton.disabled = false;
        dom.downloadButton.textContent = t('download_button_start');
        dom.cancelButton.style.display = 'none';
        dom.progressContainer.style.display = 'none';
    } else if (state === 'downloading') {
        dom.downloadButton.disabled = true;
        dom.downloadButton.textContent = t('download_button_loading');
        dom.cancelButton.style.display = 'inline-block';
        dom.resultsLog.innerHTML = '';
        dom.formError.textContent = '';
        dom.progressContainer.style.display = 'block';
        dom.progressBarInner.style.width = '0%';
        dom.progressText.textContent = t('status_starting');
    } else if (state === 'cooldown') {
        let cooldown = 3;
        dom.downloadButton.disabled = true;
        dom.cancelButton.style.display = 'none';
        const cooldownInterval = setInterval(() => {
            dom.downloadButton.textContent = t('download_button_cooldown', { seconds: cooldown });
            cooldown--;
            if (cooldown < 0) {
                clearInterval(cooldownInterval);
                setUIState('ready');
           
            }
        }, 1000);
    }
}

/**
 * Updates the UI element that lists the currently selected stations.
 * @param {Set<string>} selectedStations - A set of selected station IDs.
 * @param {object} stationsData - The main station data object.
 * @param {function} onStreamLinkClick - Callback function for when a live stream link is clicked.
 */
export function updateSelectedStationsUI(selectedStations, stationsData, onStreamLinkClick) {
    if (selectedStations.size === 0) {
        dom.stationList.style.display = 'none';
        dom.stationListPlaceholder.style.display = 'block';
    } else {
        dom.stationListPlaceholder.style.display = 'none';
        dom.stationList.style.display = 'flex';
        dom.stationList.replaceChildren(
            ...[...selectedStations].map(stationId => {
                const code = stationsData?.[stationId]?.station?.code ?? String(stationId);
                return createEl('li', { textContent: code });
            })
        );
    }
    updateLastNightButtonState(selectedStations, stationsData);
    updateLiveStreamUI(selectedStations, stationsData, onStreamLinkClick);
}

/**
 * Enables or disables the "Last Night" button based on whether a night period is calculable.
 * @param {Set<string>} selectedStations - A set of selected station IDs.
 * @param {object} stationsData - The main station data object.
 */
function updateLastNightButtonState(selectedStations, stationsData) {
    const lastNightButton = document.getElementById('last-night-btn');
    if (selectedStations.size === 0) {
        lastNightButton.disabled = true;
        return;
    }
    const firstStationId = selectedStations.values().next().value;
    const station = stationsData[firstStationId];
    const yesterday = new Date();
    yesterday.setUTCDate(yesterday.getUTCDate() - 1);
    const today = new Date();

    const yesterdayTimes = getSunTimes(yesterday, station.astronomy.latitude, station.astronomy.longitude, -6);
    const todayTimes = getSunTimes(today, station.astronomy.latitude, station.astronomy.longitude, -6);
    lastNightButton.disabled = yesterdayTimes.type === 'polar_day' || todayTimes.type === 'polar_day';
}

/**
 * Updates the live stream UI section, which is only visible when one station is selected.
 * @param {Set<string>} selectedStations - A set of selected station IDs.
 * @param {object} stationsData - The main station data object.
 * @param {function} onStreamLinkClick - Callback function for when a live stream link is clicked.
 */
function updateLiveStreamUI(selectedStations, stationsData, onStreamLinkClick) {
    if (!dom.liveStreamControls) return;
    dom.liveStreamControls.innerHTML = '';

    if (selectedStations.size === 1) {
        const stationId = selectedStations.values().next().value;
        const stationData = stationsData[stationId]; 
        const stationCode = stationData?.station?.code || 'station';
        
        const title = createEl('legend', { textContent: t('live_stream_title', { station_code: stationCode }) });
        
        // --- Video Controls ---
        const sdContainer = createEl('div', { className: 'live-stream-res-group' });
        const hdContainer = createEl('div', { className: 'live-stream-res-group' });

        for (let i = 1; i <= 7; i++) {
            const sdLink = createEl('span', { className: 'live-stream-link', textContent: `SD${i}`, onclick: () => onStreamLinkClick(stationId, i, 'lowres') });
            sdContainer.appendChild(sdLink);
            const hdLink = createEl('span', { className: 'live-stream-link', textContent: `HD${i}`, onclick: () => onStreamLinkClick(stationId, i, 'hires') });
            hdContainer.appendChild(hdLink);
        }
        
        dom.liveStreamControls.append(title, sdContainer, hdContainer);

        // --- Infrasound / Geophone Controls (New Window/Tab) ---
        const infrasoundId = stationData?.station?.infrasound_id;
        
        if (infrasoundId) {
            const sensorContainer = createEl('div', { className: 'live-stream-res-group', style: 'margin-top: 5px;' });
            
            // Infrasound Button - Opens in a new tab/window
            const infraLink = createEl('a', { // Use 'a' for better accessibility/link behavior
                href: `https://dataview.raspberryshake.org/#/AM/${infrasoundId}/00/HDF`,
                target: '_blank', // Opens in a new tab
                className: 'live-stream-link', 
                textContent: t('live_infrasound')
            });

            // Geophone Button - Opens in a new tab/window
            const geoLink = createEl('a', { // Use 'a' for better accessibility/link behavior
                href: `https://dataview.raspberryshake.org/#/AM/${infrasoundId}/00/EHZ`,
                target: '_blank', // Opens in a new tab
                className: 'live-stream-link', 
                textContent: t('live_geophone')
            });

            sensorContainer.append(infraLink, geoLink);
            dom.liveStreamControls.appendChild(sensorContainer);
        }

        dom.liveStreamControls.style.display = 'block';
    } else {
        dom.liveStreamControls.style.display = 'none';
    }
}

/**
 * Renders the list of satellite passes in the corresponding panel.
 * @param {object} passData - The data containing an array of satellite passes.
 * @param {object} callbacks - An object containing callbacks for user interactions.
 */
export function displayAllPasses(passData, { onHeaderClick, onDownloadClick, onEventClick }) {
    const satelliteList = document.getElementById('satellite-list');
    if (!passData.passes || passData.passes.length === 0) {
        satelliteList.replaceChildren(createEl('p', { style: 'color: #6c757d; margin: 0;', textContent: t('no_visible_passes') }));
        return;
    }
    satelliteList.replaceChildren();
    passData.passes.forEach((pass, index) => {
        const passDiv = createEl('div', { className: `satellite-group ${index % 2 === 0 ? 'pass-even' : 'pass-odd'}` });
        const earliestTime = new Date(pass.earliest_camera_utc);
        const formattedTimestamp = earliestTime.toISOString().slice(0, 19).replace('T', ' ');
        const header = createEl('h6', { dataset: { passId: pass.pass_id }});
        header.appendChild(document.createTextNode(t('pass_header', { satellite: pass.satellite, timestamp: formattedTimestamp }) + ' '));
        header.appendChild(createEl('span', { className: 'magnitude', textContent: t('pass_magnitude', { magnitude: pass.magnitude.toFixed(1) }) }));
        header.addEventListener('click', () => onHeaderClick(pass.pass_id, 'satellite'));

        const downloadAllBtn = createEl('button', { textContent: t('download_all_button'), className: 'download-all-btn' });
        downloadAllBtn.onclick = (e) => { e.stopPropagation(); onHeaderClick(pass.pass_id, 'satellite'); onDownloadClick(pass.pass_id, 'satellite');
        };

        const headerContainer = createEl('div', { className: 'satellite-group-header' });
        headerContainer.append(header, downloadAllBtn);
        const eventsContainer = createEl('div', { className: 'events-container' });
        pass.camera_views.forEach(event => {
            const eventSpan = createEl('span', { className: 'event-link', textContent: `${event.station_code}-${event.camera}`, dataset: { stationId: event.station_id, camera: event.camera } });
            eventSpan.addEventListener('click', () => onEventClick(pass, event));
            eventsContainer.appendChild(eventSpan);
        });
        passDiv.append(headerContainer, eventsContainer);
        satelliteList.appendChild(passDiv);
    });
}

/**
 * Renders the list of aircraft crossings in the corresponding panel.
 * @param {object} aircraftData - The data containing an array of aircraft crossings.
 * @param {object} callbacks - An object containing callbacks for user interactions.
 */
export function displayAllAircraft(aircraftData, { onHeaderClick, onDownloadClick, onEventClick }) {
    const aircraftList = document.getElementById('aircraft-list');
    const headerEl = document.querySelector('#aircraft-panel h2');
    headerEl.textContent = aircraftData.time_window_hours 
        ? t('aircraft_panel_title_dynamic', { hours: aircraftData.time_window_hours })
        : t('aircraft_panel_title');
    if (!aircraftData.crossings || aircraftData.crossings.length === 0) {
        aircraftList.replaceChildren(createEl('p', { style: 'color: #6c757d; margin: 0;', textContent: t('no_visible_aircraft') }));
        return;
    }
    aircraftList.replaceChildren();
    aircraftData.crossings.forEach((crossing, index) => {
        const crossingDiv = createEl('div', { className: `satellite-group ${index % 2 === 0 ? 'pass-even' : 'pass-odd'}` });
        const earliestTime = new Date(crossing.earliest_camera_utc);
        const formattedTimestamp = earliestTime.toISOString().slice(0, 19).replace('T', ' ');
        
        const { callsign, origin, destination } = crossing.flight_info;
        let flightIdentifier = (callsign || '????').trim();

   
         if (flightIdentifier && flightIdentifier.length > 3) {
            const icao = flightIdentifier.substring(0, 3).toUpperCase();
            const flightNumber = flightIdentifier.substring(3);
            const airlineInfo = airlineCodes[icao];

            if (airlineInfo) {
                flightIdentifier = `${airlineInfo.iata}${flightNumber} (${airlineInfo.name})`;
          
           }
        }

        const header = createEl('h6', { dataset: { crossingId: crossing.crossing_id }, textContent: t('aircraft_header', { callsign: flightIdentifier, origin: (origin || '?'), destination: (destination || '?'), timestamp: formattedTimestamp }) });
        header.addEventListener('click', () => onHeaderClick(crossing.crossing_id, 'aircraft'));

        const downloadAllBtn = createEl('button', { textContent: t('download_all_button'), className: 'download-all-btn' });
        downloadAllBtn.onclick = (e) => { e.stopPropagation(); onHeaderClick(crossing.crossing_id, 'aircraft');
        onDownloadClick(crossing.crossing_id, 'aircraft'); };

        // Add altitude quality indicator if altitude quality data is available
        const headerElements = [header, downloadAllBtn];
        if (crossing.altitude_quality) {
            const qualityIcons = { high: '📡', medium: '📶', low: '⚠️' };
            const qualityTitles = {
                high: t('altitude_quality_high'),
                medium: t('altitude_quality_medium'),
                low: t('altitude_quality_low')
            };
            const qualityIcon = createEl('span', {
                className: 'altitude-quality-indicator',
                textContent: qualityIcons[crossing.altitude_quality] || '',
                title: qualityTitles[crossing.altitude_quality] || '',
                style: 'margin-left: 8px; cursor: help;'
            });
            headerElements.splice(1, 0, qualityIcon);
        }

        const headerContainer = createEl('div', { className: 'satellite-group-header' });
        headerContainer.append(...headerElements);
        const eventsContainer = createEl('div', { className: 'events-container' });
        crossing.camera_views.forEach(event => {
            const eventSpan = createEl('span', { className: 'event-link', textContent: `${event.station_code}-${event.camera}`, dataset: { stationId: event.station_id, camera: event.camera } });
            eventSpan.addEventListener('click', () => onEventClick(crossing, event));
            eventsContainer.appendChild(eventSpan);
        });
        crossingDiv.append(headerContainer, eventsContainer);
        aircraftList.appendChild(crossingDiv);
    });
}

/**
 * Renders the list of lightning strikes in the corresponding panel.
 * @param {Array<object>} strikes - The lightning data array.
 * @param {object} stationInfo - The main station data object.
 * @param {object} cameraFovs - The camera field of view data.
 * @param {boolean} is24hFilter - Whether to filter for the last 24 hours only.
 * @param {function} onStrikeClick - Callback for when a strike list item is clicked.
 */
export function displayLightningStrikes(strikes, stationInfo, cameraFovs, is24hFilter, onStrikeClick, sortBy = 'time', subSortBy = 'time') {
    const lightningList = document.getElementById('lightning-list');
    
    // Store the last selected strike to re-apply highlighting
    const lastSelectedStrike = window.lastSelectedLightningStrike;
    document.querySelector('#lightning-panel h2').textContent = is24hFilter ? t('lightning_panel_title_24h') : t('lightning_panel_title');

    let filteredStrikes = strikes || [];
    if (is24hFilter && strikes) {
        const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
        filteredStrikes = strikes.filter(strike => new Date(strike.time) >= twentyFourHoursAgo);
    }

    if (!filteredStrikes || filteredStrikes.length === 0) {
        lightningList.replaceChildren(createEl('p', { style: 'color: #6c757d; margin: 0;', textContent: t('no_lightning_strikes') }));
        return;
    }

    // Group strikes by timestamp, station, and type
    const groupedStrikes = {};
    filteredStrikes.forEach(strike => {
        const timestamp = new Date(strike.time).toISOString().slice(0, 19).replace('T', ' ');
        const nearestStation = Object.values(stationInfo).reduce((prev, curr) => 
            L.latLng(strike.lat, strike.lon).distanceTo(L.latLng(prev.astronomy.latitude, prev.astronomy.longitude)) < 
            L.latLng(strike.lat, strike.lon).distanceTo(L.latLng(curr.astronomy.latitude, curr.astronomy.longitude)) ? prev : curr
        );
        
        const stationCode = nearestStation ? nearestStation.station.code : 'Unknown';
        const strikeTypeText = strike.type === 'cg' ? t('lightning_type_cg') : t('lightning_type_ic');
        const groupKey = `${timestamp}|${stationCode}|${strikeTypeText}`;
        
        if (!groupedStrikes[groupKey]) {
            groupedStrikes[groupKey] = {
                timestamp,
                stationCode,
                station: nearestStation,
                type: strike.type,
                typeText: strikeTypeText,
                strikes: []
            };
        }
        groupedStrikes[groupKey].strikes.push(strike);
    });

    // Sort grouped strikes based on sortBy parameter
    const sortedGroups = Object.values(groupedStrikes);
    if (sortBy === 'time') {
        sortedGroups.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    } else if (sortBy === 'station') {
        // Primary sort by station code
        sortedGroups.sort((a, b) => a.stationCode.localeCompare(b.stationCode));
        // Secondary sort within each station group
        if (subSortBy === 'time') {
            sortedGroups.sort((a, b) => {
                if (a.stationCode !== b.stationCode) return 0;
                return new Date(b.timestamp) - new Date(a.timestamp);
            });
        } else if (subSortBy === 'distance') {
            sortedGroups.sort((a, b) => {
                if (a.stationCode !== b.stationCode) return 0;
                const minDistA = Math.min(...a.strikes.map(s => s.dist));
                const minDistB = Math.min(...b.strikes.map(s => s.dist));
                return minDistA - minDistB;
            });
        }
    } else if (sortBy === 'distance') {
        sortedGroups.sort((a, b) => {
            const minDistA = Math.min(...a.strikes.map(s => s.dist));
            const minDistB = Math.min(...b.strikes.map(s => s.dist));
            return minDistA - minDistB;
        });
    }

    lightningList.replaceChildren();
    const ul = createEl('ul', { className: 'lightning-list' });
    
    sortedGroups.forEach((group, groupIndex) => {
        // Sort strikes in group by distance
        group.strikes.sort((a, b) => a.dist - b.dist);
        
        // Collect all unique camera numbers
        const allCams = new Set();
        group.strikes.forEach(strike => {
            const inViewCams = getCamerasInView(group.station, strike, cameraFovs);
            inViewCams.forEach(cam => allCams.add(cam));
        });
        
        // Create arrays for distances and bearings in the same order
        const distances = group.strikes.map(s => s.dist.toFixed(1));
        const bearings = group.strikes.map(s => {
            const bearing = calculateBearing(group.station.astronomy.latitude, group.station.astronomy.longitude, s.lat, s.lon);
            return Math.round(bearing);
        });
        
        // Sort cameras numerically
        const sortedCams = Array.from(allCams).sort((a, b) => parseInt(a) - parseInt(b));
        
        const li = createEl('li', { id: `lightning-group-${groupIndex}` });
        li.appendChild(createEl('span', { className: `strike-type-indicator ${group.type}`, textContent: '⚡' }));
        
        let stationText = '';
        if (group.station) {
            const params = {
                station_code: group.stationCode,
                dist: distances.join(', '),
                bearing: bearings.map(b => b + '°').join(', '),
                type: group.typeText,
                cams: sortedCams.join(', ')
            };
            
            if (sortedCams.length > 0) {
                stationText = t('lightning_list_item_station_info_grouped', params);
            } else {
                stationText = t('lightning_list_item_station_info_no_cam_grouped', params);
            }
        }
        
        li.appendChild(document.createTextNode(` ${group.timestamp} ${stationText}`));
        
        // Create a custom strike object for grouped strikes with all cameras
        li.onclick = () => {
            // Use the closest strike as the base, but ensure all cameras are checked
            const baseStrike = group.strikes[0];
            // Create a modified strike that will trigger checking all cameras in the group
            const groupedStrike = {
                ...baseStrike,
                isGrouped: true,
                allCams: sortedCams,
                originalStrikes: group.strikes,
                station: group.station, // Add station information for map centering
                id: `lightning-group-${groupIndex}` // Use the list item ID for highlighting
            };
            
            // Store the selected strike immediately before any display refreshes
            window.lastSelectedLightningStrike = groupedStrike;
            
            onStrikeClick(groupedStrike, true);
        };
        ul.appendChild(li);
    });
    
    lightningList.appendChild(ul);
    
    // Re-apply highlighting if there was a previously selected strike
    if (lastSelectedStrike) {
        setTimeout(() => {
            // Trigger the selection again to re-apply highlighting
            if (typeof onStrikeClick === 'function') {
                onStrikeClick(lastSelectedStrike, false);
            }
        }, 100);
    }
}


/**
 * Creates and displays a modal window containing an Iframe for external data (Infrasound/Geophone).
 * @param {string} url - The URL to load.
 * @param {string} title - The title for the modal.
 */
function showIframeModal(url, title) {
    const modalBackdrop = createEl('div', { id: 'video-modal-backdrop' });
    const modalContent = createEl('div', { id: 'video-modal-content' });
    
    // Create a container for the iframe
    const iframeContainer = createEl('div', { 
        id: 'video-container', 
        style: 'aspect-ratio: 16/9; background: #fff;' 
    });

    const iframe = createEl('iframe', { 
        src: url, 
        style: 'width: 100%; height: 100%; border: none;',
        allowfullscreen: true
    });

    const statusEl = createEl('p', { id: 'video-status', textContent: title });
    const closeButton = createEl('button', { 
        id: 'video-close-button', 
        textContent: t('modal_close_button'), 
        onclick: () => document.getElementById('video-modal-backdrop')?.remove() 
    });

    iframeContainer.appendChild(iframe);
    modalContent.append(statusEl, iframeContainer, closeButton);
    modalBackdrop.appendChild(modalContent);
    document.body.appendChild(modalBackdrop);
}

/**
 * Creates and displays a video preview player with frame controls, filters, and screenshot capability.
 * @param {string} videoUrl - The URL of the video to preview.
 * @param {string} title - The title for the modal.
 * @param {Array} mediaList - Optional list of all media items for navigation.
 * @param {number} mediaIndex - Optional index of current item in mediaList.
 * @param {Object} initialDimensions - Optional {width, height} to use until content loads.
 */
export function showVideoPreview(videoUrl, title, mediaList = null, mediaIndex = -1, initialDimensions = null) {
    const mediaInfo = mediaList && mediaIndex >= 0 ? mediaList[mediaIndex] : null;
    const knownDuration = (mediaInfo && typeof mediaInfo.duration === 'number' && isFinite(mediaInfo.duration))
        ? mediaInfo.duration : null;
    const knownStartTime = (mediaInfo && typeof mediaInfo.start_time === 'number' && isFinite(mediaInfo.start_time))
        ? mediaInfo.start_time : null;

    // Variables used for absolute-timeline detection (also needed by dynamic overlays).
    let useAbsoluteTime = false;
    let absoluteStartTime = 0;

    const modalBackdrop = createEl('div', { id: 'video-modal-backdrop' });
    const modalContent = createEl('div', { id: 'video-modal-content', className: 'preview-modal' });

    // Apply initial dimensions if provided (for smooth navigation)
    if (initialDimensions) {
        // Disable transitions temporarily to prevent visible resize
        modalContent.style.transition = 'none';
        modalContent.style.width = initialDimensions.width + 'px';
        modalContent.style.height = initialDimensions.height + 'px';
        modalContent.style.minWidth = 'auto';
        modalContent.style.minHeight = 'auto';
        // Re-enable transitions after a delay
        setTimeout(() => { modalContent.style.transition = ''; }, 50);
    }

    // Build enhanced title from filename
    const enhancedTitle = buildEnhancedPreviewTitle(title, videoUrl);

    // Header with title and close button
    const header = createEl('div', { className: 'preview-header' });
    header.appendChild(createEl('h3', { textContent: enhancedTitle, className: 'preview-title' }));
    const closeButton = createEl('button', { className: 'preview-close-btn', textContent: '×' });
    header.appendChild(closeButton);

    const isHighResTimelapse = /_(teqh|tfeh)\.mp4$/i.test(title);

    // Video container with overlay for timestamp
    const videoWrapper = createEl('div', { className: 'preview-video-wrapper' });
    const isMultiMinuteVideo = /_dur\d+_/.test(title);
    const video = createEl('video', {
        src: videoUrl,
        className: 'preview-video',
        controls: false,
        preload: isMultiMinuteVideo ? 'auto' : 'metadata',
        autoplay: !isHighResTimelapse,
        muted: true
    });

    // Grid, annotation, and camera-boundary overlays for archive videos (hidden by default via opacity, shown via toggle)
    const gridOverlay = createEl('img', { id: 'grid-overlay-image', className: 'archive-overlay grid-overlay', style: { display: 'block', position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 10 } });
    const annotationOverlay = createEl('img', { id: 'annotation-overlay-image', className: 'archive-overlay annotation-overlay', style: { display: 'block', position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 11 } });
    const boundsOverlay = createEl('img', { id: 'bounds-overlay-image', className: 'archive-overlay bounds-overlay', style: { display: 'block', position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 13 } });
    // Mask overlay: black=sky made transparent, white=foreground made opaque
    // black by the backend. Drawn above grid/annotations but below the
    // camera-boundary overlay, so "Vis kameragrenser" stays visible on top
    // of the mask.
    const maskOverlay = createEl('img', { id: 'mask-overlay-image', className: 'archive-overlay mask-overlay', style: { display: 'block', position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 12 } });
    // Explicitly set opacity 0 to hide initially
    gridOverlay.style.opacity = '0';
    annotationOverlay.style.opacity = '0';
    boundsOverlay.style.opacity = '0';
    maskOverlay.style.opacity = '0';

    // Timestamp overlay with date (2 decimal precision) - lower right
    const timestampOverlay = createEl('div', { className: 'preview-timestamp', textContent: '' });

    // Loading indicator and error overlay
    const loadingIndicator = createEl('div', { className: 'preview-loading', textContent: t('loading') });
    const errorOverlay = createEl('div', {
        className: 'preview-error-overlay',
        style: {
            display: 'none',
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            background: 'rgba(0,0,0,0.85)',
            color: '#fff',
            justifyContent: 'center',
            alignItems: 'center',
            textAlign: 'center',
            padding: '20px',
            boxSizing: 'border-box',
            zIndex: 20,
            pointerEvents: 'none'
        }
    });

    videoWrapper.append(video, gridOverlay, boundsOverlay, annotationOverlay, maskOverlay, timestampOverlay, loadingIndicator, errorOverlay);

    // Playback controls
    const controls = createEl('div', { className: 'preview-controls' });

    // Play/Pause button - video autoplays unless high-res timelapse is blocked
    const playPauseBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: isHighResTimelapse ? '▶' : '⏸',
        title: t('modal_play_pause')
    });

    // Frame step controls
    const frameBackBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '◀',
        title: t('modal_frame_back')
    });

    const frameForwardBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '▶',
        title: t('modal_frame_forward')
    });

    // Rewind button
    const rewindBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '⏮',
        title: t('modal_rewind')
    });

    // Screenshot button
    const screenshotBtn = createEl('button', {
        className: 'preview-control-btn screenshot',
        textContent: '📷',
        title: t('screenshot')
    });

    // Download button
    const downloadBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '⬇',
        title: t('download_video')
    });

    // Fullscreen button
    const fullscreenBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '⛶',
        title: t('fullscreen')
    });

    // Navigation buttons (prev/next) - shown when mediaList is provided
    // Always create buttons even with 1 item, since more may load later
    // Buttons not disabled - click handlers check bounds dynamically using currentMediaList
    let prevBtn = null, nextBtn = null, navInfo = null;
    if (mediaList && mediaList.length > 0) {
        prevBtn = createEl('button', {
            className: 'preview-control-btn nav-btn',
            textContent: '◀',
            title: t('previous')
        });
        nextBtn = createEl('button', {
            className: 'preview-control-btn nav-btn',
            textContent: '▶',
            title: t('next')
        });
        // Use currentMediaList for dynamic total count
        const totalCount = currentMediaList.length || mediaList.length;
        navInfo = createEl('span', {
            className: 'nav-info',
            textContent: `${mediaIndex + 1} / ${totalCount}`,
            style: { fontSize: '12px', color: '#8aa4be', margin: '0 8px' }
        });
    }

    if (prevBtn && nextBtn) {
        controls.append(prevBtn, navInfo, nextBtn, rewindBtn, frameBackBtn, playPauseBtn, frameForwardBtn, screenshotBtn, downloadBtn, fullscreenBtn);
    } else {
        controls.append(rewindBtn, frameBackBtn, playPauseBtn, frameForwardBtn, screenshotBtn, downloadBtn, fullscreenBtn);
    }

    // Scrubber bar
    const scrubberRow = createEl('div', { className: 'preview-scrubber-row' });
    const scrubberTime = createEl('span', { className: 'preview-scrubber-time', textContent: '0:00' });
    const scrubberDur  = createEl('span', { className: 'preview-scrubber-time', textContent: '0:00' });
    const scrubber = createEl('input', {
        type: 'range', min: '0', max: '100', step: '0.1', value: '0',
        className: 'preview-scrubber'
    });
    scrubberRow.append(scrubberTime, scrubber);

    // Seed the scrubber with the known duration immediately so it does not
    // start short and then grow while the browser refines its metadata.
    if (knownDuration !== null && isFinite(knownDuration) && knownDuration > 0) {
        scrubber.max = knownDuration;
        if (knownStartTime && knownStartTime > 1e9) {
            scrubberDur.textContent = getFormattedTimestamp(knownStartTime + knownDuration);
        } else {
            scrubberDur.textContent = fmtTime(knownDuration);
        }
    }

    // Filter controls - all on one line
    const filterControls = createEl('div', { className: 'preview-filter-controls' });

    // Brightness slider
    const brightnessSlider = createEl('input', {
        type: 'range',
        min: '0.5',
        max: '2',
        step: '0.1',
        value: '1',
        className: 'preview-slider',
        title: t('brightness'),
        id: 'brightness-slider'
    });

    // Contrast slider
    const contrastSlider = createEl('input', {
        type: 'range',
        min: '0.5',
        max: '2',
        step: '0.1',
        value: '1',
        className: 'preview-slider',
        title: t('contrast'),
        id: 'contrast-slider'
    });

    // Saturation slider
    const saturationSlider = createEl('input', {
        type: 'range',
        min: '0',
        max: '3',
        step: '0.1',
        value: '1',
        className: 'preview-slider',
        title: t('saturation'),
        id: 'saturation-slider'
    });

    // Speed slider
    const speedSlider = createEl('input', {
        type: 'range',
        min: '0.1',
        max: '4',
        step: '0.05',
        value: '1',
        className: 'preview-slider',
        title: t('playback_speed'),
        id: 'speed-slider'
    });
    const speedLabel = createEl('label', { htmlFor: 'speed-slider', className: 'preview-filter-label', style: { whiteSpace: 'nowrap' } });
    const updateSpeedLabel = (rate) => { speedLabel.textContent = `${t('playback_speed')} ${rate.toFixed(2).replace(/\.?0+$/, '')}\u00d7`; };
    updateSpeedLabel(1);

    // Reset filters button
    const resetFiltersBtn = createEl('button', {
        className: 'preview-control-btn reset',
        textContent: t('reset_filters'),
        title: t('reset_filters')
    });

    // Timestamp toggle checkbox
    const timestampToggleContainer = createEl('label', { className: 'preview-timestamp-toggle' });
    const timestampCheckbox = createEl('input', {
        type: 'checkbox',
        checked: false
    });
    timestampToggleContainer.append(timestampCheckbox, ' ', t('show_timestamp'));

    // Detect timelapse files: STATION_YYYYMMDD_teq.mp4, _tfe.mp4, _teqh.mp4, _tfeh.mp4
    const timelapseFisheye = title.match(/^([A-Z]{2,4})_(\d{8})_tfeh?\.mp4$/);
    const timelapseEquirect = title.match(/^([A-Z]{2,4})_(\d{8})_teqh?\.mp4$/);
    const timelapseFull = timelapseFisheye || timelapseEquirect;
    // Equirectangular timelapses are 360°-wide: dragging sideways should roll/wrap
    // the video seamlessly instead of the normal clamped pan behaviour.
    const isEquirectVideo = !!timelapseEquirect;

    // Parse station and camera from video filename (e.g., "GAU_cam1_20260429_2056_hires.mp4")
    const filenameMatch = title.match(/^([A-Z]{3})_cam(\d+)_\d{8}_\d{4}/);
    let stationId = null, cameraNum = null, videoTimestamp = null, annotationTimestamp = null;
    let gridToggleContainer = null, annotationToggleContainer = null, boundsToggleContainer = null, maskToggleContainer = null;
    let gridCheckbox = null, annotationCheckbox = null, boundsCheckbox = null, maskCheckbox = null;

    if (timelapseFull) {
        // Timelapse: derive stationId, cameraNum, timestamp from filename
        stationId = timelapseFull[1];
        cameraNum = timelapseFisheye ? '9' : '8';
        const ds = timelapseFull[2]; // YYYYMMDD
        videoTimestamp = `${ds.substring(0,4)}-${ds.substring(4,6)}-${ds.substring(6,8)}T00:00:00`;

        // Grid overlay toggle only (no annotation for timelapse)
        gridToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        gridCheckbox = createEl('input', { type: 'checkbox', id: 'grid-overlay-toggle', disabled: true });
        gridToggleContainer.append(gridCheckbox, ' ', t('modal_grid_toggle'));

        // Camera boundary overlay toggle
        boundsToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        boundsCheckbox = createEl('input', { type: 'checkbox', id: 'bounds-overlay-toggle', disabled: true });
        boundsToggleContainer.append(boundsCheckbox, ' ', t('modal_bounds_toggle'));

        // Mask overlay toggle (cam8/cam9 stitched masks)
        maskToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        maskCheckbox = createEl('input', { type: 'checkbox', id: 'mask-overlay-toggle', disabled: true });
        maskToggleContainer.append(maskCheckbox, ' ', t('modal_mask_toggle'));
    } else if (filenameMatch) {
        stationId = filenameMatch[1];
        cameraNum = filenameMatch[2];

        // Extract timestamp from filename for overlays
        const dateMatch = title.match(/(\d{8})_(\d{4})/);
        if (dateMatch) {
            const dateStr = dateMatch[1];
            const timeStr = dateMatch[2];
            const year = dateStr.substring(0, 4);
            const month = dateStr.substring(4, 6);
            const day = dateStr.substring(6, 8);
            const hour = timeStr.substring(0, 2);
            const minute = timeStr.substring(2, 4);
            videoTimestamp = `${year}-${month}-${day}T${hour}:${minute}:00`;
            // Add 30 seconds for annotation (middle of video)
            annotationTimestamp = `${year}-${month}-${day}T${hour}:${minute}:30`;
        }

        // Grid overlay toggle - initially greyed out until loaded
        gridToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        gridCheckbox = createEl('input', { type: 'checkbox', id: 'grid-overlay-toggle', disabled: true });
        gridToggleContainer.append(gridCheckbox, ' ', t('modal_grid_toggle'));

        // Annotation overlay toggle - initially greyed out until loaded
        annotationToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        annotationCheckbox = createEl('input', { type: 'checkbox', id: 'annotation-overlay-toggle', disabled: true });
        annotationToggleContainer.append(annotationCheckbox, ' ', t('modal_annotation_toggle'));

        // Mask overlay toggle - initially greyed out until loaded
        maskToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        maskCheckbox = createEl('input', { type: 'checkbox', id: 'mask-overlay-toggle', disabled: true });
        maskToggleContainer.append(maskCheckbox, ' ', t('modal_mask_toggle'));
    }

    // Brightness: label above slider in a small inline column
    const brightnessWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    brightnessWrapper.append(
        createEl('label', { textContent: t('brightness'), htmlFor: 'brightness-slider', className: 'preview-filter-label' }),
        brightnessSlider
    );

    // Contrast: label above slider in a small inline column
    const contrastWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    contrastWrapper.append(
        createEl('label', { textContent: t('contrast'), htmlFor: 'contrast-slider', className: 'preview-filter-label' }),
        contrastSlider
    );

    // Saturation: label above slider in a small inline column
    const saturationWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    saturationWrapper.append(
        createEl('label', { textContent: t('saturation'), htmlFor: 'saturation-slider', className: 'preview-filter-label' }),
        saturationSlider
    );

    const checkboxesWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '4px' } });
    if (!timelapseFull) checkboxesWrapper.append(timestampToggleContainer);
    if (gridToggleContainer) checkboxesWrapper.append(gridToggleContainer);
    if (boundsToggleContainer) checkboxesWrapper.append(boundsToggleContainer);
    if (annotationToggleContainer) checkboxesWrapper.append(annotationToggleContainer);
    if (maskToggleContainer) checkboxesWrapper.append(maskToggleContainer);

    // Speed: combined label+value above slider
    const speedWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    speedWrapper.append(speedLabel, speedSlider);

    filterControls.append(resetFiltersBtn, speedWrapper, brightnessWrapper, contrastWrapper, saturationWrapper, checkboxesWrapper);

    // Assemble modal. Timelapse videos skip the title header (the title
    // adds little value there and the extra height isn't worth it), but
    // still need a visible close button, so move it out of the header and
    // float it over the top-right corner of the video instead.
    if (timelapseFull) {
        closeButton.classList.add('preview-close-btn-overlay');
        videoWrapper.appendChild(closeButton);
        modalContent.append(videoWrapper, scrubberRow, controls, filterControls);
    } else {
        modalContent.append(header, videoWrapper, scrubberRow, controls, filterControls);
    }
    modalBackdrop.appendChild(modalContent);
    document.body.appendChild(modalBackdrop);

    // Push history state for back button handling
    history.pushState({ modalOpen: true }, '');

    // Load grid overlay - fetch JSON metadata first, then set image src
    if (videoTimestamp && stationId && cameraNum) {
        // For timelapse, use the local stitch grid (same as fisheye/equirect images)
        const gridApiUrl = timelapseFull
            ? `index.php?action=fetch_stitch_grid&projection=${timelapseFisheye ? 'fe' : 'eq'}&resolution=hires`
            : `index.php?action=fetch_archive_grid&station_id=${stationId}&camera_num=${cameraNum}&timestamp=${encodeURIComponent(videoTimestamp)}`;
        fetch(gridApiUrl)
            .then(response => response.json())
            .then(data => {
                if (data.success && data.grid_url) {
                    gridOverlay.src = data.grid_url;
                    gridToggleContainer.style.opacity = '1';
                    gridCheckbox.disabled = false;
                } else {
                    gridToggleContainer.style.opacity = '0.5';
                    gridCheckbox.disabled = true;
                }
            })
            .catch(() => {
                gridToggleContainer.style.opacity = '0.5';
                gridCheckbox.disabled = true;
            });
        gridCheckbox.addEventListener('change', () => {
            gridOverlay.style.opacity = gridCheckbox.checked ? '0.6' : '0';
        });

        if (timelapseFull && boundsToggleContainer) {
            const proj = timelapseFisheye ? 'fe' : 'eq';
            fetch(`index.php?action=fetch_stitch_cam_boundaries&station_id=${stationId}&projection=${proj}&resolution=lowres`)
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.grid_url) {
                        boundsOverlay.src = data.grid_url;
                        boundsToggleContainer.style.opacity = '1';
                        boundsCheckbox.disabled = false;
                    }
                })
                .catch(() => {});
            boundsCheckbox.addEventListener('change', () => {
                boundsOverlay.style.opacity = boundsCheckbox.checked ? '0.8' : '0';
            });
        }

        // Load mask overlay - fetch JSON metadata first, then set image src.
        // Works for every camera number, including the stitched cam8/cam9.
        if (maskToggleContainer) {
            fetch(`index.php?action=fetch_archive_mask&station_id=${stationId}&camera_num=${cameraNum}`)
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.mask_url) {
                        maskOverlay.src = data.mask_url;
                        maskToggleContainer.style.opacity = '1';
                        maskCheckbox.disabled = false;
                    }
                })
                .catch(() => {});
            maskCheckbox.addEventListener('change', () => {
                maskOverlay.style.opacity = maskCheckbox.checked ? '1' : '0';
            });
        }

        if (!timelapseFull) {
        // Load annotation overlay - fetch JSON metadata first, then set image src
        function setAnnotationOverlay(url) {
            if (url && annotationOverlay.src !== url) annotationOverlay.src = url;
        }

        if (isMultiMinuteVideo && knownStartTime && knownDuration && knownDuration > 60) {
            // Fetch the per-minute star annotations in the background so playback can
            // start immediately, then enable the toggle as soon as the overlays are ready.
            annotationToggleContainer.style.opacity = '0.5';
            annotationCheckbox.disabled = true;
            let minuteUrls = null;

            const loadMinuteOverlays = async () => {
                if (minuteUrls) return;
                const minuteCount = Math.max(1, Math.ceil(knownDuration / 60));
                const startMs = knownStartTime * 1000;
                const promises = [];
                for (let i = 0; i < minuteCount; i++) {
                    const ts = new Date(startMs + i * 60000 + 30000);
                    const tsStr = ts.toISOString().slice(0, 19);
                    const url = `index.php?action=fetch_archive_annotation&station_id=${stationId}&camera_num=${cameraNum}&timestamp=${encodeURIComponent(tsStr)}`;
                    promises.push(
                        fetch(url)
                            .then(r => r.json())
                            .then(d => (d.success && d.annotation_url) ? d.annotation_url : null)
                            .catch(() => null)
                    );
                }
                const urls = await Promise.all(promises);
                if (urls.some(Boolean)) {
                    minuteUrls = urls;
                    annotationToggleContainer.style.opacity = '1';
                    annotationCheckbox.disabled = false;
                    if (annotationCheckbox.checked) updateAnnotationOverlay();
                }
                // If no overlays are available, leave the control greyed out/disabled.
            };

            loadMinuteOverlays();

            const updateAnnotationOverlay = () => {
                if (!annotationCheckbox.checked || !minuteUrls) return;
                const relTime = Math.max(0, useAbsoluteTime ? video.currentTime - absoluteStartTime : video.currentTime);
                const idx = Math.min(minuteUrls.length - 1, Math.max(0, Math.floor(relTime / 60)));
                setAnnotationOverlay(minuteUrls[idx]);
            };

            video.addEventListener('timeupdate', updateAnnotationOverlay);
            annotationCheckbox.addEventListener('change', () => {
                annotationOverlay.style.opacity = annotationCheckbox.checked && minuteUrls ? '0.6' : '0';
                updateAnnotationOverlay();
            });
        } else {
            const annotationApiUrl = `index.php?action=fetch_archive_annotation&station_id=${stationId}&camera_num=${cameraNum}&timestamp=${encodeURIComponent(annotationTimestamp)}`;
            fetch(annotationApiUrl)
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.annotation_url) {
                        setAnnotationOverlay(data.annotation_url);
                        annotationToggleContainer.style.opacity = '1';
                        annotationCheckbox.disabled = false;
                    } else {
                        annotationToggleContainer.style.opacity = '0.5';
                        annotationCheckbox.disabled = true;
                    }
                })
                .catch(err => {
                    annotationToggleContainer.style.opacity = '0.5';
                    annotationCheckbox.disabled = true;
                });

            // Annotation toggle handler - toggle opacity (0.6)
            annotationCheckbox.addEventListener('change', () => {
                annotationOverlay.style.opacity = annotationCheckbox.checked ? '0.6' : '0';
            });
        }
        } // end if (!timelapseFull)
    }

    // Video event handlers
    let isPlaying = !isHighResTimelapse; // Video autoplays unless high-res timelapse is blocked
    let frameStep = 1 / 30; // Assume 30fps, will be updated when metadata loads

    // Base epoch from filename timestamp (e.g. 2026-07-10T22:42:00 UTC)
    const videoBaseEpochMs = videoTimestamp ? Date.parse(videoTimestamp + 'Z') : null;
    // Helper: format a UTC Date as "YYYY-MM-DD HH:MM:SS.cc"
    function formatUtcTimestamp(dateMs, subSecOffset) {
        const d = new Date(dateMs);
        const pad = (n, w=2) => String(n).padStart(w, '0');
        const decimals = String(Math.floor((subSecOffset % 1) * 100)).padStart(2, '0');
        return `${d.getUTCFullYear()}-${pad(d.getUTCMonth()+1)}-${pad(d.getUTCDate())} ` +
               `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}.${decimals}`;
    }

    // Helper to format timestamp overlay: base epoch + playback offset
    function getFormattedTimestamp(seconds) {
        const effectiveSeconds = seconds || 0;
        // Safety net: if the video timeline is Unix-epoch based, treat it as absolute.
        if (useAbsoluteTime || effectiveSeconds > 1e9) {
            // Media timeline itself is Unix-epoch seconds (e.g. raw station video).
            return formatUtcTimestamp(effectiveSeconds * 1000, effectiveSeconds);
        }
        if (videoBaseEpochMs !== null) {
            return formatUtcTimestamp(videoBaseEpochMs + effectiveSeconds * 1000, effectiveSeconds);
        }
        // Fallback: no base epoch, show playback offset only
        const pad = (n) => String(n).padStart(2, '0');
        const total = Math.floor(effectiveSeconds);
        const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
        const decimals = String(Math.floor((effectiveSeconds % 1) * 100)).padStart(2, '0');
        return `${pad(h)}:${pad(m)}:${pad(s)}.${decimals}`;
    }

    // Set initial timestamp immediately before video loads (hidden for timelapse)
    if (timelapseFull) {
        timestampOverlay.style.display = 'none';
    } else {
        timestampOverlay.textContent = getFormattedTimestamp(0);
    }

    video.addEventListener('loadedmetadata', () => {
        loadingIndicator.style.display = 'none';
        errorOverlay.style.display = 'none';
        // Clear fixed dimensions to allow natural video sizing
        if (initialDimensions) {
            modalContent.style.width = '';
            modalContent.style.height = '';
            modalContent.style.minWidth = '';
            modalContent.style.minHeight = '';
        }
        // Try to detect frame rate from video or default to 30
        frameStep = 1 / 30;
        // Station videos often use Unix-epoch timestamps as their media timeline.
        // Detect that so the overlay shows the actual UTC time, not filename time.
        ensureAbsoluteTime();
        // Trigger timeupdate to refresh timestamp overlay at the current start time
        video.dispatchEvent(new Event('timeupdate'));
    });

    video.addEventListener('error', () => {
        const code = video.error?.code;
        // Transient decode errors often fire while the user is scrubbing rapidly.
        // Don't show the error overlay during an active scrub; the change handler
        // will seek to the final position and any real failure will surface then.
        if (scrubbing && (code === MediaError.MEDIA_ERR_DECODE || code === MediaError.MEDIA_ERR_NETWORK)) {
            return;
        }
        loadingIndicator.style.display = 'none';
        errorOverlay.style.display = 'flex';
        if (isHighResTimelapse && (code === 3 || code === 4)) {
            errorOverlay.textContent = t('video_error_highres', {
                width: video.videoWidth || 4096,
                height: video.videoHeight || 4096
            });
        } else {
            errorOverlay.textContent = t('video_error_generic') + (video.error?.message ? ` ${video.error.message}` : '');
        }
    });

    // Very high-resolution H.264 timelapse files (e.g. 4096×4096) exceed common
    // browser decoder limits and just play as a black screen. Block autoplay until
    // metadata is available, then warn only if the decoded dimensions are too large.
    let highResWarningTimer = null;
    let highResMetadataLoaded = false;
    if (isHighResTimelapse) {
        const showHighResWarning = (w, h) => {
            if (errorOverlay.style.display === 'flex') return;
            loadingIndicator.style.display = 'none';
            errorOverlay.style.display = 'flex';
            errorOverlay.textContent = t('video_error_highres', {
                width: w || video.videoWidth || 4096,
                height: h || video.videoHeight || 4096
            });
            isPlaying = false;
            playPauseBtn.textContent = '▶';
            if (highResWarningTimer) {
                clearTimeout(highResWarningTimer);
                highResWarningTimer = null;
            }
        };
        video.addEventListener('loadedmetadata', () => {
            highResMetadataLoaded = true;
            const width = video.videoWidth;
            const height = video.videoHeight;
            if (width > 4096 || height > 2304) {
                showHighResWarning(width, height);
            } else {
                video.play();
                isPlaying = true;
                playPauseBtn.textContent = '⏸';
            }
        }, { once: true });
        highResWarningTimer = setTimeout(() => {
            if (errorOverlay.style.display !== 'flex' && !highResMetadataLoaded) {
                showHighResWarning(4096, 4096);
            }
        }, 1500);
    }

    video.addEventListener('timeupdate', () => {
        // Update timestamp overlay with date and 2-decimal precision (lower right)
        if (timelapseFull) return;
        ensureAbsoluteTime();
        if (timestampCheckbox.checked) {
            timestampOverlay.textContent = getFormattedTimestamp(video.currentTime);
            timestampOverlay.style.display = 'block';
        } else {
            timestampOverlay.style.display = 'none';
        }
    });

    video.addEventListener('ended', () => {
        isPlaying = false;
        playPauseBtn.textContent = '▶';
    });

    // Stall / decode-too-slow recovery: step down playbackRate on repeated stalls
    const RATE_STEPS = [1.0, 0.75, 0.5, 0.25];
    let rateStepIdx = 0;
    let stallTimer = null;
    const stallStatusEl = createEl('div', { className: 'preview-stall-status', style: { fontSize: '11px', color: '#f0c040', textAlign: 'center', display: 'none', padding: '2px 0' } });
    // Insert just above the controls
    modalContent.insertBefore(stallStatusEl, controls);

    const applyRateStep = () => {
        if (rateStepIdx >= RATE_STEPS.length - 1) return; // already at minimum
        rateStepIdx++;
        const newRate = RATE_STEPS[rateStepIdx];
        video.playbackRate = newRate;
        speedSlider.value = newRate;
        updateSpeedLabel(newRate);
        stallStatusEl.textContent = `${t('playback_slowed')} ${Math.round(newRate * 100)}%`;
        stallStatusEl.style.display = 'block';
    };

    const onStall = () => {
        // Debounce: only act if video has been stalled for >1.5 s
        if (stallTimer) return;
        stallTimer = setTimeout(() => {
            stallTimer = null;
            if (!video.paused && video.readyState < 3) applyRateStep();
        }, 1500);
    };
    const clearStall = () => { if (stallTimer) { clearTimeout(stallTimer); stallTimer = null; } };

    video.addEventListener('stalled', onStall);
    video.addEventListener('waiting', onStall);
    video.addEventListener('playing', clearStall);
    video.addEventListener('canplay', clearStall);

    video.addEventListener('error', () => {
        const err = video.error;
        // code 1 = MEDIA_ERR_ABORTED (seek/navigation abort) — not a real failure
        // null error = spurious event fired during buffering; ignore both
        if (!err || err.code === MediaError.MEDIA_ERR_ABORTED) return;
        loadingIndicator.textContent = t('video_load_error');
        loadingIndicator.style.display = 'block';
        loadingIndicator.style.color = '#f87';
    });

    // Control handlers
    playPauseBtn.addEventListener('click', () => {
        if (isPlaying) {
            video.pause();
            playPauseBtn.textContent = '▶';
        } else {
            // If playback has already ended, explicitly seek back to the
            // start so the scrubber and video both reset on replay.
            if (video.ended) {
                video.currentTime = absoluteStartTime;
            }
            video.play();
            playPauseBtn.textContent = '⏸';
        }
        isPlaying = !isPlaying;
    });

    frameBackBtn.addEventListener('click', () => {
        video.pause();
        isPlaying = false;
        playPauseBtn.textContent = '▶';
        video.currentTime = Math.max(absoluteStartTime, video.currentTime - frameStep);
    });

    frameForwardBtn.addEventListener('click', () => {
        video.pause();
        isPlaying = false;
        playPauseBtn.textContent = '▶';
        video.currentTime = Math.min(absoluteStartTime + getVideoDuration(), video.currentTime + frameStep);
    });

    // Rewind button handler
    rewindBtn.addEventListener('click', () => {
        video.currentTime = absoluteStartTime;
        if (!isPlaying) {
            video.play();
            isPlaying = true;
            playPauseBtn.textContent = '⏸';
        }
    });

    // Scrubber handlers
    function fmtTime(s) {
        if (!isFinite(s)) return '0:00';
        const m = Math.floor(s / 60), sec = Math.floor(s % 60);
        return `${m}:${sec.toString().padStart(2, '0')}`;
    }
    let scrubbing = false, wasPlayingBeforeScrub = false;
    let clipDuration = null, clipEndTime = null;

    // Compute the real clip duration/end once. The browser may update buffered ranges
    // as it decodes, so never derive duration from buffered ranges.
    const computeClipBounds = () => {
        if (clipDuration !== null) return; // already computed
        ensureAbsoluteTime();
        let dur = null;
        if (knownDuration !== null && isFinite(knownDuration) && knownDuration > 0) {
            dur = knownDuration;
        } else if (useAbsoluteTime && absoluteStartTime > 1e9 && video.duration > absoluteStartTime) {
            // For Unix-epoch-timeline videos the browser reports the absolute end time.
            dur = video.duration - absoluteStartTime;
        } else {
            dur = video.duration;
        }
        if (!isFinite(dur) || dur <= 0) dur = 0;
        clipDuration = dur;
        clipEndTime = useAbsoluteTime ? absoluteStartTime + dur : dur;
    };

    video.addEventListener('loadedmetadata', () => {
        computeClipBounds();
        if (isFinite(clipDuration)) {
            scrubber.max = clipDuration;
            scrubberDur.textContent = useAbsoluteTime
                ? getFormattedTimestamp(clipEndTime)
                : fmtTime(clipDuration);
        }
    });
    // Some videos (especially downloaded MP4s with the moov atom at the end)
    // report their duration later than loadedmetadata; update when it changes.
    video.addEventListener('durationchange', () => {
        // Force recomputation so an initial short duration is corrected.
        clipDuration = null;
        clipEndTime = null;
        computeClipBounds();
        if (isFinite(clipDuration)) {
            scrubber.max = clipDuration;
            scrubberDur.textContent = useAbsoluteTime
                ? getFormattedTimestamp(clipEndTime)
                : fmtTime(clipDuration);
            updateScrubber();
        }
    });
    // Detect absolute timeline lazily if loadedmetadata didn't catch it.
    // Once detected, the start time must stay fixed so the scrubber doesn't drift
    // as playback progresses.
    let absoluteTimeDetected = false;
    const ensureAbsoluteTime = () => {
        if (absoluteTimeDetected) return;
        if (knownStartTime !== null) {
            useAbsoluteTime = true;
            absoluteStartTime = knownStartTime;
            absoluteTimeDetected = true;
            return;
        }
        let start = null;
        if (video.seekable.length > 0) {
            start = video.seekable.start(0);
        } else if (video.buffered.length > 0) {
            start = video.buffered.start(0);
        }
        if (start !== null) {
            if (start > 1e9) {
                useAbsoluteTime = true;
                absoluteStartTime = start;
            } else {
                useAbsoluteTime = false;
                absoluteStartTime = 0;
            }
            absoluteTimeDetected = true;
        }
    };

    const getVideoDuration = () => {
        if (clipDuration === null) computeClipBounds();
        return clipDuration;
    };

    const updateScrubber = () => {
        const dur = getVideoDuration();
        if (!scrubbing && isFinite(dur)) {
            ensureAbsoluteTime();
            const relativeTime = Math.max(0, video.currentTime - absoluteStartTime);
            scrubber.value = relativeTime;
            scrubberTime.textContent = useAbsoluteTime
                ? getFormattedTimestamp(video.currentTime)
                : fmtTime(relativeTime);
        }
    };
    video.addEventListener('timeupdate', updateScrubber);
    // Ensure the scrubber snaps to the correct position on explicit seeks
    // (rewind, frame step, replay after ended) in addition to normal playback.
    video.addEventListener('seeked', updateScrubber);
    // Fallback animation-frame loop for players/browsers that fire timeupdate
    // too sparsely (e.g. some HEVC / long-GOP files) so the thumb still moves.
    let scrubberRafId = null;
    const scrubberFrameLoop = () => {
        updateScrubber();
        if (!video.paused && !video.ended) {
            scrubberRafId = requestAnimationFrame(scrubberFrameLoop);
        } else {
            scrubberRafId = null;
        }
    };
    video.addEventListener('play', () => {
        if (!scrubberRafId) scrubberRafId = requestAnimationFrame(scrubberFrameLoop);
    });
    video.addEventListener('pause', () => {
        if (scrubberRafId) {
            cancelAnimationFrame(scrubberRafId);
            scrubberRafId = null;
        }
        updateScrubber();
    });
    video.addEventListener('ended', () => {
        if (scrubberRafId) {
            cancelAnimationFrame(scrubberRafId);
            scrubberRafId = null;
        }
        updateScrubber();
    });
    let scrubStartCurrentTime = 0, scrubStartScrubberValue = 0;
    // When the absolute start_time wasn't supplied/detected, derive the base
    // from the currentTime and scrubber position at the start of the drag.
    const getScrubberBase = () => {
        if (absoluteStartTime > 1e9) return absoluteStartTime;
        if (scrubStartCurrentTime > 1e9) {
            return scrubStartCurrentTime - scrubStartScrubberValue;
        }
        return 0;
    };
    // Adaptive scrubber seeking: update the timestamp immediately, but queue
    // video seeks so rapid dragging never overloads the decoder with overlapping
    // seeks. Once the current seek finishes, the latest requested target is applied.
    let pendingSeekTarget = null;
    let isSeekInProgress = false;
    let seekSafetyTimer = null;
    const onScrubberSeekDone = () => {
        isSeekInProgress = false;
        if (seekSafetyTimer) {
            clearTimeout(seekSafetyTimer);
            seekSafetyTimer = null;
        }
        if (pendingSeekTarget !== null) {
            const t = pendingSeekTarget;
            pendingSeekTarget = null;
            scheduleScrubberSeek(t);
        }
    };
    const scheduleScrubberSeek = (targetTime) => {
        if (isSeekInProgress) {
            pendingSeekTarget = targetTime;
            return;
        }
        isSeekInProgress = true;
        pendingSeekTarget = null;
        video.currentTime = targetTime;
        seekSafetyTimer = setTimeout(onScrubberSeekDone, 500);
    };
    video.addEventListener('seeked', onScrubberSeekDone);

    scrubber.addEventListener('mousedown', () => {
        scrubbing = true;
        wasPlayingBeforeScrub = isPlaying;
        scrubStartCurrentTime = video.currentTime;
        scrubStartScrubberValue = parseFloat(scrubber.value);
        video.pause();
    });
    scrubber.addEventListener('input', () => {
        const relativeTime = parseFloat(scrubber.value);
        const targetTime = getScrubberBase() + relativeTime;
        scrubberTime.textContent = useAbsoluteTime
            ? getFormattedTimestamp(targetTime)
            : fmtTime(relativeTime);
        scheduleScrubberSeek(targetTime);
    });
    scrubber.addEventListener('change', () => {
        scrubbing = false;
        const relativeTime = parseFloat(scrubber.value);
        const targetTime = getScrubberBase() + relativeTime;
        pendingSeekTarget = null;
        scheduleScrubberSeek(targetTime);
        if (wasPlayingBeforeScrub) {
            video.play();
        } else {
            isPlaying = false;
            playPauseBtn.textContent = '▶';
        }
    });

    // Filter handlers
    function updateFilters() {
        const brightness = brightnessSlider.value;
        const contrast = contrastSlider.value;
        const saturation = saturationSlider.value;
        video.style.filter = `brightness(${brightness}) contrast(${contrast}) saturate(${saturation})`;
    }

    brightnessSlider.addEventListener('input', updateFilters);
    contrastSlider.addEventListener('input', updateFilters);
    saturationSlider.addEventListener('input', updateFilters);

    // Speed slider handler — manual adjustment also freezes auto-reduction
    speedSlider.addEventListener('input', () => {
        const rate = parseFloat(speedSlider.value);
        video.playbackRate = rate;
        updateSpeedLabel(rate);
        // Sync stall-reduction index so auto-steps continue from current rate
        rateStepIdx = RATE_STEPS.reduce((best, r, i) => Math.abs(r - rate) < Math.abs(RATE_STEPS[best] - rate) ? i : best, 0);
    });

    // Reset filters button handler
    resetFiltersBtn.addEventListener('click', () => {
        brightnessSlider.value = 1;
        contrastSlider.value = 1;
        saturationSlider.value = 1;
        updateFilters();
        speedSlider.value = 1;
        updateSpeedLabel(1);
        video.playbackRate = 1;
        rateStepIdx = 0;
        stallStatusEl.style.display = 'none';
    });

    // Timestamp toggle handler
    timestampCheckbox.addEventListener('change', () => {
        const event = new Event('timeupdate');
        video.dispatchEvent(event);
    });

    // Fullscreen button handler - use videoWrapper to include overlays
    fullscreenBtn.addEventListener('click', () => {
        if (videoWrapper.requestFullscreen) {
            videoWrapper.requestFullscreen();
        } else if (videoWrapper.webkitRequestFullscreen) {
            videoWrapper.webkitRequestFullscreen();
        } else if (videoWrapper.msRequestFullscreen) {
            videoWrapper.msRequestFullscreen();
        }
    });

    // Spacebar toggles play/pause while in fullscreen
    const onFullscreenKeydown = (e) => {
        if (e.defaultPrevented) return;
        if ((e.key === ' ' || e.code === 'Space') && document.fullscreenElement === videoWrapper) {
            e.preventDefault();
            playPauseBtn.click();
        }
    };
    document.addEventListener('keydown', onFullscreenKeydown);

    // Sync overlays to the actual rendered video area (letterboxed inside the wrapper)
    // Also sets transformOrigin to content centre for correct zoom behaviour.
    // contentRect holds {x, y, w, h} relative to videoWrapper top-left.
    let contentRect = { x: 0, y: 0, w: 0, h: 0 };
    // Assigned later (equirect only) to keep the roll canvases' geometry in sync.
    let rollGeometrySync = null;
    // Assigned later (equirect only); mirrors video's transformOrigin so zoom centres correctly.
    let rollViewport = null;
    const syncVideoOverlays = () => {
        const vw = video.offsetWidth, vh = video.offsetHeight;
        const nw = video.videoWidth || vw, nh = video.videoHeight || vh;
        if (!nw || !nh) return;
        const videoAspect = nw / nh, boxAspect = vw / vh;
        let rw, rh, rl, rt;
        if (videoAspect > boxAspect) {
            rw = vw; rh = vw / videoAspect;
            rl = video.offsetLeft; rt = video.offsetTop + (vh - rh) / 2;
        } else {
            rh = vh; rw = vh * videoAspect;
            rt = video.offsetTop; rl = video.offsetLeft + (vw - rw) / 2;
        }
        contentRect = { x: rl, y: rt, w: rw, h: rh };
        // transformOrigin on content centre so scale() zooms from the middle
        const originX = (rl + rw / 2) + 'px';
        const originY = (rt + rh / 2) + 'px';
        video.style.transformOrigin = originX + ' ' + originY;
        if (rollViewport) rollViewport.style.transformOrigin = originX + ' ' + originY;
        [gridOverlay, boundsOverlay, annotationOverlay, maskOverlay].forEach(ov => {
            ov.style.left            = rl + 'px';
            ov.style.top             = rt + 'px';
            ov.style.width           = rw + 'px';
            ov.style.height          = rh + 'px';
            ov.style.transformOrigin = (rw / 2) + 'px ' + (rh / 2) + 'px';
        });
        if (rollGeometrySync) rollGeometrySync();
    };
    video.addEventListener('loadedmetadata', () => {
        if (video.videoWidth && video.videoHeight) {
            videoWrapper.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
        }
        syncVideoOverlays();
    });
    const videoRo = new ResizeObserver(() => setTimeout(syncVideoOverlays, 0));
    videoRo.observe(videoWrapper);

    makeModalResizable(modalContent, syncVideoOverlays);

    // --- Pan/Zoom and Fullscreen Scrub Logic ---
    let scale=1, panX=0, panY=0, isPanning=false, startPanX=0, startPanY=0, panOriginX=0, panOriginY=0;
    let scrubDragActive=false, scrubStartX=0, scrubStartTime=0, scrubDidMove=false;
    let scrubDragRaf=null, scrubDragMoveX=null, lastScrubDx=0, wasPlayingBeforeFullscreenScrub=false;
    // Equirectangular roll state: horizontal offset is applied to the roll canvases
    // (see below) so it wraps seamlessly instead of clamping like panX.
    let rollOffsetPx = 0, startRollOffset = 0, rollRafId = null;
    const clamp = (val, min, max) => Math.min(Math.max(val, min), max);
    const updateTransform = () => { 
        const transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
        video.style.transform = gridOverlay.style.transform = boundsOverlay.style.transform = annotationOverlay.style.transform = maskOverlay.style.transform = transform;
        if (rollViewport) rollViewport.style.transform = `translate(0px, ${panY}px) scale(${scale})`;
    };
    const getContentCentre = () => {
        const rect = videoWrapper.getBoundingClientRect();
        return { cx: rect.left + contentRect.x + contentRect.w / 2,
                 cy: rect.top  + contentRect.y + contentRect.h / 2 };
    };
    const onWheel = e => {
        e.preventDefault();
        const { cx, cy } = getContentCentre();
        const mouseX = e.clientX - cx, mouseY = e.clientY - cy;
        const newScale = clamp(scale * (e.deltaY > 0 ? 0.9 : 1.1), 1, 8);
        const newPanX = mouseX - (mouseX - panX) * (newScale / scale);
        const newPanY = mouseY - (mouseY - panY) * (newScale / scale);
        scale = newScale;
        if (scale <= 1.01) { panX = 0; panY = 0; } else {
            const maxPanX = contentRect.w * (scale - 1) / 2;
            const maxPanY = contentRect.h * (scale - 1) / 2;
            panX = clamp(newPanX, -maxPanX, maxPanX);
            panY = clamp(newPanY, -maxPanY, maxPanY);
        }
        updateTransform();
    };
    const onMouseMove = e => {
        if (scrubDragActive) {
            scrubDragMoveX = e.clientX;
            if (scrubDragRaf) return;
            scrubDragRaf = requestAnimationFrame(() => {
                scrubDragRaf = null;
                if (!scrubDragActive) return;
                const dx = scrubDragMoveX - scrubStartX;
                if (!scrubDidMove && Math.abs(dx) > 5) scrubDidMove = true;
                if (scrubDidMove && Math.abs(dx - lastScrubDx) >= 10) {
                    const duration = video.duration || 0;
                    const width = videoWrapper.clientWidth || window.innerWidth;
                    const sensitivity = (duration > 0 && width > 0) ? duration / width : 0.1;
                    const targetTime = clamp(scrubStartTime + dx * sensitivity, 0, duration);
                    lastScrubDx = dx;
                    scheduleScrubberSeek(targetTime);
                }
            });
            return;
        }
        if (!isPanning) return;
        const maxPanY = contentRect.h * (scale - 1) / 2;
        panY = clamp(startPanY + (e.clientY - panOriginY), -maxPanY, maxPanY);
        if (isEquirectVideo) {
            rollOffsetPx = startRollOffset + (e.clientX - panOriginX) / scale;
            if (rollGeometrySync) rollGeometrySync();
        } else {
            const maxPanX = contentRect.w * (scale - 1) / 2;
            panX = clamp(startPanX + (e.clientX - panOriginX), -maxPanX, maxPanX);
        }
        updateTransform();
    };
    const onMouseUp = () => {
        if (scrubDragRaf) {
            cancelAnimationFrame(scrubDragRaf);
            scrubDragRaf = null;
        }
        if (scrubDragActive) {
            const wasClick = !scrubDidMove;
            if (!wasClick && scrubDragMoveX !== null) {
                const dx = scrubDragMoveX - scrubStartX;
                const duration = video.duration || 0;
                const width = videoWrapper.clientWidth || window.innerWidth;
                const sensitivity = (duration > 0 && width > 0) ? duration / width : 0.1;
                const targetTime = clamp(scrubStartTime + dx * sensitivity, 0, duration);
                pendingSeekTarget = null;
                scheduleScrubberSeek(targetTime);
            }
            scrubDragActive = false;
            scrubDidMove = false;
            scrubDragMoveX = null;
            videoWrapper.style.cursor = 'default';
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
            if (wasClick) {
                playPauseBtn.click();
            } else if (wasPlayingBeforeFullscreenScrub) {
                video.play();
            } else {
                isPlaying = false;
                playPauseBtn.textContent = '▶';
            }
            return;
        }
        if (!isPanning) return;
        isPanning = false;
        videoWrapper.style.cursor = isEquirectVideo ? 'grab' : 'default';
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
    };
    const onMouseDown = e => {
        if (e.button !== 0) return;
        if (!isEquirectVideo && document.fullscreenElement === videoWrapper && scale <= 1.01) {
            e.preventDefault();
            scrubDragActive = true;
            scrubStartX = e.clientX;
            scrubStartTime = video.currentTime;
            scrubDidMove = false;
            scrubDragMoveX = e.clientX;
            lastScrubDx = 0;
            wasPlayingBeforeFullscreenScrub = isPlaying;
            if (isPlaying) video.pause();
            videoWrapper.style.cursor = 'ew-resize';
            window.addEventListener('mousemove', onMouseMove);
            window.addEventListener('mouseup', onMouseUp);
            return;
        }
        e.preventDefault();
        isPanning = true;
        videoWrapper.style.cursor = 'grabbing';
        panOriginX = e.clientX;
        panOriginY = e.clientY;
        startPanX = panX;
        startPanY = panY;
        startRollOffset = rollOffsetPx;
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
    };
    videoWrapper.addEventListener('wheel', onWheel); videoWrapper.addEventListener('mousedown', onMouseDown);

    if (isEquirectVideo) {
        // Roll canvas mirrors the (now-hidden) video + grid/bounds/mask overlays by
        // repainting the current video frame + overlays every animation frame, drawn
        // twice at a wrapping horizontal offset (exactly like CSS background-repeat-x
        // with background-position-x) so dragging rolls the 360° panorama seamlessly.
        // A single canvas (rather than a second <video>) avoids decoding the source
        // twice and stays perfectly in sync with the one playing <video>.
        video.style.visibility = 'hidden';
        gridOverlay.style.visibility = 'hidden';
        boundsOverlay.style.visibility = 'hidden';
        maskOverlay.style.visibility = 'hidden';

        rollViewport = createEl('canvas', { style: { position: 'absolute', pointerEvents: 'none', zIndex: 1 } });
        videoWrapper.appendChild(rollViewport);
        videoWrapper.style.cursor = 'grab';

        const applyRollGeometry = () => {
            const { x: rl, y: rt, w: rw, h: rh } = contentRect;
            if (!rw || !rh) return;
            rollViewport.style.left = rl + 'px';
            rollViewport.style.top = rt + 'px';
            rollViewport.style.width = rw + 'px';
            rollViewport.style.height = rh + 'px';
            const cw = Math.max(1, Math.round(rw)), ch = Math.max(1, Math.round(rh));
            if (rollViewport.width !== cw) rollViewport.width = cw;
            if (rollViewport.height !== ch) rollViewport.height = ch;
        };
        rollGeometrySync = applyRollGeometry;

        const ctx = rollViewport.getContext('2d');
        const drawLayer = (dx, srcImg, w, h) => {
            const opacity = parseFloat(srcImg.style.opacity) || 0;
            if (opacity <= 0 || !srcImg.complete || !srcImg.naturalWidth) return;
            ctx.globalAlpha = opacity;
            ctx.drawImage(srcImg, dx, 0, w, h);
            ctx.globalAlpha = 1;
        };
        const drawFrame = () => {
            const w = rollViewport.width, h = rollViewport.height;
            if (w && h) {
                ctx.clearRect(0, 0, w, h);
                // Positive rollOffsetPx shifts content right, same convention as the
                // image viewer's background-position-x drag handling. Snapped to a
                // whole pixel: the image viewer's CSS background-repeat tiling is
                // seamless by construction, but two canvas drawImage() calls abutting
                // at a sub-pixel x get edge-antialiased independently, which visibly
                // blends the (slightly different) colours on each side of the wrap
                // into a thin seam. Rounding removes that sub-pixel blend entirely.
                let mod = rollOffsetPx % w;
                if (mod < 0) mod += w;
                mod = Math.round(mod);
                const hasVideoFrame = video.readyState >= 2 && video.videoWidth && video.videoHeight;
                for (const dx of [mod, mod - w]) {
                    if (hasVideoFrame) ctx.drawImage(video, dx, 0, w, h);
                    drawLayer(dx, gridOverlay, w, h);
                    drawLayer(dx, maskOverlay, w, h);
                    drawLayer(dx, boundsOverlay, w, h);
                }
            }
            rollRafId = requestAnimationFrame(drawFrame);
        };
        rollRafId = requestAnimationFrame(drawFrame);

        applyRollGeometry();
    }

    // Sync overlay visibility when entering/exiting fullscreen
    const onFullscreenChange = () => {
        const isFullscreen = !!document.fullscreenElement;
        if (isFullscreen) {
            video.style.maxHeight = 'none';
            video.style.height = '100%';
        } else {
            video.style.maxHeight = '';
            video.style.height = '';
            scale=1; panX=0; panY=0; updateTransform();
            scrubDragActive = false;
            isPanning = false;
            videoWrapper.style.cursor = 'default';
        }
        setTimeout(syncVideoOverlays, 50);
        gridOverlay.style.opacity = gridCheckbox?.checked ? '0.6' : '0';
        boundsOverlay.style.opacity = boundsCheckbox?.checked ? '0.8' : '0';
        annotationOverlay.style.opacity = annotationCheckbox?.checked ? '0.6' : '0';
        maskOverlay.style.opacity = maskCheckbox?.checked ? '1' : '0';
    };
    document.addEventListener('fullscreenchange', onFullscreenChange);

    // Screenshot handler
    screenshotBtn.addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');

        // Apply current filters to canvas
        ctx.filter = video.style.filter;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Draw grid overlay if enabled
        if (gridCheckbox?.checked && gridOverlay.src) {
            ctx.globalAlpha = 0.6;
            ctx.drawImage(gridOverlay, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
        }

        // Draw annotation overlay if enabled
        if (annotationCheckbox?.checked && annotationOverlay.src) {
            ctx.globalAlpha = 0.6;
            ctx.drawImage(annotationOverlay, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
        }

        // Draw mask overlay if enabled - its own alpha channel already
        // encodes sky (transparent) vs. foreground (opaque black), so no
        // extra globalAlpha blending is needed. Drawn before the camera
        // bounds overlay so "Vis kameragrenser" stays visible on top.
        if (maskCheckbox?.checked && maskOverlay.src) {
            ctx.drawImage(maskOverlay, 0, 0, canvas.width, canvas.height);
        }

        // Draw camera bounds overlay if enabled
        if (boundsCheckbox?.checked && boundsOverlay.src) {
            ctx.globalAlpha = 0.8;
            ctx.drawImage(boundsOverlay, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
        }

        // Add timestamp to screenshot only if enabled (lower right corner)
        if (timestampCheckbox.checked) {
            ctx.filter = 'none';
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            // Measure text to fit box exactly
            ctx.font = '16px monospace';
            const text = timestampOverlay.textContent;
            const textWidth = ctx.measureText(text).width;
            const padding = 10;
            const x = canvas.width - textWidth - padding * 2 - 10;
            const y = canvas.height - 38;
            // Draw box in lower right, sized to fit text
            ctx.fillRect(x, y, textWidth + padding * 2, 28);
            ctx.fillStyle = '#fff';
            ctx.fillText(text, x + padding, y + 20);
        }

        // Download screenshot
        const link = document.createElement('a');
        link.download = `screenshot_${title.replace(/[^a-z0-9]/gi, '_')}_${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    });

    // Download handler
    downloadBtn.addEventListener('click', () => {
        const link = document.createElement('a');
        link.href = videoUrl;
        link.download = title;
        link.click();
    });

    // Navigation button handlers - use currentMediaList for dynamic bounds checking
    if (prevBtn && nextBtn && mediaList && mediaIndex >= 0) {
        prevBtn.addEventListener('click', () => {
            if (mediaIndex > 0) {
                const prevItem = currentMediaList[mediaIndex - 1];
                // Capture current dimensions for smooth transition
                const rect = modalContent.getBoundingClientRect();
                lastModalDimensions = { width: rect.width, height: rect.height };
                closeButton.click();
                if (prevItem.isVideo) {
                    showVideoPreview(prevItem.url, prevItem.name, currentMediaList, mediaIndex - 1, lastModalDimensions);
                } else {
                    showImagePreview(prevItem.url, prevItem.name, currentMediaList, mediaIndex - 1, lastModalDimensions);
                }
            }
        });
        nextBtn.addEventListener('click', () => {
            if (mediaIndex < currentMediaList.length - 1) {
                const nextItem = currentMediaList[mediaIndex + 1];
                // Capture current dimensions for smooth transition
                const rect = modalContent.getBoundingClientRect();
                lastModalDimensions = { width: rect.width, height: rect.height };
                closeButton.click();
                if (nextItem.isVideo) {
                    showVideoPreview(nextItem.url, nextItem.name, currentMediaList, mediaIndex + 1, lastModalDimensions);
                } else {
                    showImagePreview(nextItem.url, nextItem.name, currentMediaList, mediaIndex + 1, lastModalDimensions);
                }
            }
        });
    }

    // Keyboard shortcuts - check bounds dynamically instead of using disabled state
    modalBackdrop.addEventListener('keydown', (e) => {
        switch(e.key) {
            case 'ArrowLeft':
                e.preventDefault();
                if (e.shiftKey && prevBtn && mediaIndex > 0) {
                    prevBtn.click();
                } else {
                    frameBackBtn.click();
                }
                break;
            case 'ArrowRight':
                e.preventDefault();
                if (e.shiftKey && nextBtn && mediaIndex < currentMediaList.length - 1) {
                    nextBtn.click();
                } else {
                    frameForwardBtn.click();
                }
                break;
            case ' ':
                e.preventDefault();
                playPauseBtn.click();
                break;
            case 'Escape':
                e.preventDefault();
                closeButton.click();
                break;
        }
    });

    // Close button handler (defined after all handlers)
    closeButton.addEventListener('click', () => {
        if (highResWarningTimer) clearTimeout(highResWarningTimer);
        document.removeEventListener('fullscreenchange', onFullscreenChange);
        document.removeEventListener('keydown', onFullscreenKeydown);
        videoWrapper.removeEventListener('wheel', onWheel);
        videoWrapper.removeEventListener('mousedown', onMouseDown);
        videoRo.disconnect();
        if (rollRafId) cancelAnimationFrame(rollRafId);
        history.back();
    });

    // Focus modal for keyboard events
    modalBackdrop.setAttribute('tabindex', '0');
    setTimeout(() => modalBackdrop.focus(), 100);
}

/**
 * Creates and displays a modal for viewing a downloaded image with brightness/contrast/zoom.
 * @param {string} imageUrl - The URL of the image to preview.
 * @param {string} title - The title for the modal.
 * @param {Array} mediaList - Optional list of all media items for navigation.
 * @param {number} mediaIndex - Optional index of current item in mediaList.
 * @param {Object} initialDimensions - Optional {width, height} to use until content loads.
 */
export function showImagePreview(imageUrl, title, mediaList = null, mediaIndex = -1, initialDimensions = null) {
    const modalBackdrop = createEl('div', { id: 'video-modal-backdrop' });
    const modalContent = createEl('div', { id: 'video-modal-content', className: 'preview-modal' });

    // Apply initial dimensions if provided (for smooth navigation)
    if (initialDimensions) {
        // Disable transitions temporarily to prevent visible resize
        modalContent.style.transition = 'none';
        modalContent.style.width = initialDimensions.width + 'px';
        modalContent.style.height = initialDimensions.height + 'px';
        modalContent.style.minWidth = 'auto';
        modalContent.style.minHeight = 'auto';
        // Re-enable transitions after a delay
        setTimeout(() => { modalContent.style.transition = ''; }, 50);
    }

    // Build enhanced title from filename
    const enhancedTitle = buildEnhancedPreviewTitle(title, imageUrl);

    // Header
    const header = createEl('div', { className: 'preview-header' });
    header.appendChild(createEl('h3', { textContent: enhancedTitle, className: 'preview-title' }));
    const closeButton = createEl('button', { className: 'preview-close-btn', textContent: '×' });
    header.appendChild(closeButton);

    // Image wrapper with pan/zoom
    const imageWrapper = createEl('div', { className: 'preview-video-wrapper' });
    const img = createEl('img', { src: imageUrl, className: 'preview-video' });

    // Overlays: positioned to exactly match img's rendered position/size within wrapper
    const gridOverlay = createEl('img', { className: 'archive-overlay grid-overlay', style: { display: 'block', position: 'absolute', pointerEvents: 'none', zIndex: 10, opacity: '0' } });
    const boundsOverlay = createEl('img', { className: 'archive-overlay bounds-overlay', style: { display: 'block', position: 'absolute', pointerEvents: 'none', zIndex: 13, opacity: '0' } });
    const annotationOverlay = createEl('img', { className: 'archive-overlay annotation-overlay', style: { display: 'block', position: 'absolute', pointerEvents: 'none', zIndex: 11, opacity: '0' } });
    // Mask overlay: black=sky made transparent, white=foreground made opaque
    // black by the backend. Drawn above grid/annotations but below the
    // camera-boundary overlay, so "Vis kameragrenser" stays visible on top
    // of the mask.
    const maskOverlay = createEl('img', { className: 'archive-overlay mask-overlay', style: { display: 'block', position: 'absolute', pointerEvents: 'none', zIndex: 12, opacity: '0' } });

    const loadingIndicator = createEl('div', { className: 'preview-loading', textContent: t('loading') });
    imageWrapper.append(img, gridOverlay, boundsOverlay, annotationOverlay, maskOverlay, loadingIndicator);

    img.addEventListener('load', () => {
        loadingIndicator.style.display = 'none';
        // Clear fixed dimensions to allow natural image sizing
        if (initialDimensions) {
            modalContent.style.width = '';
            modalContent.style.height = '';
            modalContent.style.minWidth = '';
            modalContent.style.minHeight = '';
        }
    });

    // Controls
    const controls = createEl('div', { className: 'preview-controls' });

    const downloadBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '⬇',
        title: t('download_image')
    });
    const fullscreenBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '⛶',
        title: t('fullscreen')
    });

    // Navigation buttons (prev/next) - shown when mediaList is provided
    // Always create buttons even with 1 item, since more may load later
    // Buttons not disabled - click handlers check bounds dynamically using currentMediaList
    let prevBtn = null, nextBtn = null, navInfo = null;
    if (mediaList && mediaList.length > 0) {
        prevBtn = createEl('button', {
            className: 'preview-control-btn nav-btn',
            textContent: '◀',
            title: t('previous')
        });
        nextBtn = createEl('button', {
            className: 'preview-control-btn nav-btn',
            textContent: '▶',
            title: t('next')
        });
        // Use currentMediaList for dynamic total count
        const totalCount = currentMediaList.length || mediaList.length;
        navInfo = createEl('span', {
            className: 'nav-info',
            textContent: `${mediaIndex + 1} / ${totalCount}`,
            style: { fontSize: '12px', color: '#8aa4be', margin: '0 8px' }
        });
    }

    if (prevBtn && nextBtn) {
        controls.append(prevBtn, navInfo, nextBtn, fullscreenBtn, downloadBtn);
    } else {
        controls.append(fullscreenBtn, downloadBtn);
    }

    // Filter controls
    const filterControls = createEl('div', { className: 'preview-filter-controls' });
    const brightnessSlider = createEl('input', { type: 'range', min: '0.5', max: '2', step: '0.1', value: '1', className: 'preview-slider', id: 'img-brightness-slider' });
    const contrastSlider = createEl('input', { type: 'range', min: '0.5', max: '2', step: '0.1', value: '1', className: 'preview-slider', id: 'img-contrast-slider' });
    const saturationSlider = createEl('input', { type: 'range', min: '0', max: '3', step: '0.1', value: '1', className: 'preview-slider', id: 'img-saturation-slider' });
    const resetFiltersBtn = createEl('button', { className: 'preview-control-btn reset', textContent: t('reset_filters'), title: t('reset_filters') });

    const brightnessWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    brightnessWrapper.append(createEl('label', { textContent: t('brightness'), htmlFor: 'img-brightness-slider', className: 'preview-filter-label' }), brightnessSlider);
    const contrastWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    contrastWrapper.append(createEl('label', { textContent: t('contrast'), htmlFor: 'img-contrast-slider', className: 'preview-filter-label' }), contrastSlider);
    const saturationWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    saturationWrapper.append(createEl('label', { textContent: t('saturation'), htmlFor: 'img-saturation-slider', className: 'preview-filter-label' }), saturationSlider);

    // Parse station and camera from image filename (e.g., "GAU_cam1_20260429_2056_image.jpg")
    const filenameMatch = title.match(/^([A-Z]{3})_cam(\d+)_(\d{8})_(\d{4})/);
    // Detect stitched panorama filenames (e.g. "GAU_20260429_2056_hires_equirect.jpg" or "..._hires_long_equirect.jpg")
    // Match against imageUrl since title may be a short display name like 'eqh'
    const stitchMatch = imageUrl.match(/_(hires|lowres)(?:_long)?_(equirect|fisheye)\.jpg(?:[?#].*)?$/i);
    // Equirectangular panoramas are 360°-wide: dragging sideways should roll/wrap
    // the image seamlessly instead of the normal clamped pan behaviour.
    const isEquirect = !!(stitchMatch && stitchMatch[2].toLowerCase() === 'equirect');
    let gridToggleContainer = null, annotationToggleContainer = null, boundsToggleContainer = null, maskToggleContainer = null;
    let gridCheckbox = null, annotationCheckbox = null, boundsCheckbox = null, maskCheckbox = null;

    if (stitchMatch) {
        const resolution = stitchMatch[1].toLowerCase();   // 'hires' or 'lowres'
        const projection = stitchMatch[2].toLowerCase() === 'equirect' ? 'eq' : 'fe';

        gridToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        gridCheckbox = createEl('input', { type: 'checkbox', id: 'img-grid-overlay-toggle', disabled: true });
        gridToggleContainer.append(gridCheckbox, ' ', t('modal_grid_toggle'));

        boundsToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        boundsCheckbox = createEl('input', { type: 'checkbox', id: 'img-bounds-overlay-toggle', disabled: true });
        boundsToggleContainer.append(boundsCheckbox, ' ', t('modal_bounds_toggle'));

        maskToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        maskCheckbox = createEl('input', { type: 'checkbox', id: 'img-mask-overlay-toggle', disabled: true });
        maskToggleContainer.append(maskCheckbox, ' ', t('modal_mask_toggle'));

        fetch(`index.php?action=fetch_stitch_grid&projection=${projection}&resolution=${resolution}`)
            .then(r => r.json())
            .then(data => {
                if (data.success && data.grid_url) {
                    gridOverlay.src = data.grid_url;
                    gridToggleContainer.style.opacity = '1';
                    gridCheckbox.disabled = false;
                }
            })
            .catch(() => {});

        // Extract station code from the URL (e.g. GAU_20260429_2056_hires_equirect.jpg)
        const stationCodeMatch = imageUrl.match(/\/([A-Z]{2,4})_\d{8}_/);
        if (stationCodeMatch) {
            fetch(`index.php?action=fetch_stitch_cam_boundaries&station_id=${stationCodeMatch[1]}&projection=${projection}&resolution=${resolution}`)
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.grid_url) {
                        boundsOverlay.src = data.grid_url;
                        boundsToggleContainer.style.opacity = '1';
                        boundsCheckbox.disabled = false;
                    }
                })
                .catch(() => {});

            // Stitched panoramas expose their mask as cam8 (equirect) / cam9 (fisheye)
            const maskCameraNum = projection === 'eq' ? '8' : '9';
            fetch(`index.php?action=fetch_archive_mask&station_id=${stationCodeMatch[1]}&camera_num=${maskCameraNum}`)
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.mask_url) {
                        maskOverlay.src = data.mask_url;
                        maskToggleContainer.style.opacity = '1';
                        maskCheckbox.disabled = false;
                    }
                })
                .catch(() => {});
        }

        gridCheckbox.addEventListener('change', () => {
            gridOverlay.style.opacity = gridCheckbox.checked ? '0.6' : '0';
        });
        boundsCheckbox.addEventListener('change', () => {
            boundsOverlay.style.opacity = boundsCheckbox.checked ? '0.8' : '0';
        });
        maskCheckbox.addEventListener('change', () => {
            maskOverlay.style.opacity = maskCheckbox.checked ? '1' : '0';
        });
    } else if (filenameMatch) {
        const stationId = filenameMatch[1];
        const cameraNum = filenameMatch[2];
        const dateStr = filenameMatch[3];
        const timeStr = filenameMatch[4];
        const year = dateStr.substring(0, 4);
        const month = dateStr.substring(4, 6);
        const day = dateStr.substring(6, 8);
        const hour = timeStr.substring(0, 2);
        const minute = timeStr.substring(2, 4);
        const imageTimestamp = `${year}-${month}-${day}T${hour}:${minute}:00`;
        const annotationTimestamp = `${year}-${month}-${day}T${hour}:${minute}:30`;

        // Grid overlay toggle - initially greyed out until loaded
        gridToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        gridCheckbox = createEl('input', { type: 'checkbox', id: 'img-grid-overlay-toggle', disabled: true });
        gridToggleContainer.append(gridCheckbox, ' ', t('modal_grid_toggle'));

        // Annotation overlay toggle - initially greyed out until loaded
        annotationToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        annotationCheckbox = createEl('input', { type: 'checkbox', id: 'img-annotation-overlay-toggle', disabled: true });
        annotationToggleContainer.append(annotationCheckbox, ' ', t('modal_annotation_toggle'));

        // Mask overlay toggle - initially greyed out until loaded
        maskToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        maskCheckbox = createEl('input', { type: 'checkbox', id: 'img-mask-overlay-toggle', disabled: true });
        maskToggleContainer.append(maskCheckbox, ' ', t('modal_mask_toggle'));

        // Fetch mask overlay
        fetch(`index.php?action=fetch_archive_mask&station_id=${stationId}&camera_num=${cameraNum}`)
            .then(r => r.json())
            .then(data => {
                if (data.success && data.mask_url) {
                    maskOverlay.src = data.mask_url;
                    maskToggleContainer.style.opacity = '1';
                    maskCheckbox.disabled = false;
                }
            })
            .catch(() => {});

        maskCheckbox.addEventListener('change', () => {
            maskOverlay.style.opacity = maskCheckbox.checked ? '1' : '0';
        });

        // Fetch grid overlay
        fetch(`index.php?action=fetch_archive_grid&station_id=${stationId}&camera_num=${cameraNum}&timestamp=${encodeURIComponent(imageTimestamp)}`)
            .then(r => r.json())
            .then(data => {
                if (data.success && data.grid_url) {
                    gridOverlay.src = data.grid_url;
                    gridToggleContainer.style.opacity = '1';
                    gridCheckbox.disabled = false;
                }
            })
            .catch(() => {});

        gridCheckbox.addEventListener('change', () => {
            gridOverlay.style.opacity = gridCheckbox.checked ? '0.6' : '0';
        });

        // Fetch annotation overlay
        fetch(`index.php?action=fetch_archive_annotation&station_id=${stationId}&camera_num=${cameraNum}&timestamp=${encodeURIComponent(annotationTimestamp)}`)
            .then(r => r.json())
            .then(data => {
                if (data.success && data.annotation_url) {
                    annotationOverlay.src = data.annotation_url;
                    annotationToggleContainer.style.opacity = '1';
                    annotationCheckbox.disabled = false;
                }
            })
            .catch(() => {});

        annotationCheckbox.addEventListener('change', () => {
            annotationOverlay.style.opacity = annotationCheckbox.checked ? '0.6' : '0';
        });
    }

    const checkboxesWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '4px' } });
    if (gridToggleContainer) checkboxesWrapper.append(gridToggleContainer);
    if (boundsToggleContainer) checkboxesWrapper.append(boundsToggleContainer);
    if (annotationToggleContainer) checkboxesWrapper.append(annotationToggleContainer);
    if (maskToggleContainer) checkboxesWrapper.append(maskToggleContainer);

    // Enhance filter slider
    const enhanceWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    const enhanceLabel = createEl('label', { textContent: t('filter'), htmlFor: 'enhance-slider', className: 'preview-filter-label' });
    const enhanceSlider = createEl('input', {
        type: 'range',
        min: '0',
        max: '64',
        step: '1',
        value: '0',
        className: 'preview-slider',
        id: 'enhance-slider',
        title: 'Enhance filter threshold (0 = off, 64 = max)'
    });
    enhanceWrapper.append(enhanceLabel, enhanceSlider);

    filterControls.append(resetFiltersBtn, brightnessWrapper, contrastWrapper, saturationWrapper, enhanceWrapper);
    if (checkboxesWrapper.hasChildNodes()) filterControls.append(checkboxesWrapper);

    function updateFilters() {
        img.style.filter = `brightness(${brightnessSlider.value}) contrast(${contrastSlider.value}) saturate(${saturationSlider.value})`;
    }
    brightnessSlider.addEventListener('input', updateFilters);
    contrastSlider.addEventListener('input', updateFilters);
    saturationSlider.addEventListener('input', updateFilters);
    resetFiltersBtn.addEventListener('click', () => { brightnessSlider.value = 1; contrastSlider.value = 1; saturationSlider.value = 1; enhanceSlider.value = 0; updateFilters(); resetEnhance(); });

    // Enhance filter handler
    let originalImageData = null;
    let enhanceTimeout = null;

    async function applyEnhanceFilter(threshold) {
        if (threshold == 0) {
            // Reset to original image
            if (originalImageData) {
                img.src = originalImageData;
            }
            return;
        }

        try {
            const response = await fetch(`index.php?action=enhance_filter&image=${encodeURIComponent(imageUrl)}&threshold=${threshold}`);
            const data = await response.json();
            if (data.image) {
                img.src = `data:image/jpeg;base64,${data.image}`;
            } else {
                // Empty response means use original
                if (originalImageData) {
                    img.src = originalImageData;
                }
            }
        } catch (error) {
            console.error('Error applying enhance filter:', error);
        }
    }

    function resetEnhance() {
        if (originalImageData) {
            img.src = originalImageData;
        }
    }

    // Store original image when loaded
    img.addEventListener('load', () => {
        if (!originalImageData) {
            originalImageData = imageUrl;
        }
    });

    // Debounced enhance filter application
    enhanceSlider.addEventListener('input', () => {
        const threshold = parseInt(enhanceSlider.value);
        if (enhanceTimeout) clearTimeout(enhanceTimeout);
        enhanceTimeout = setTimeout(() => applyEnhanceFilter(threshold), 300);
    });

    // Assemble modal
    modalContent.append(header, imageWrapper, controls, filterControls);
    modalBackdrop.appendChild(modalContent);
    document.body.appendChild(modalBackdrop);

    // Push history state for back button handling
    history.pushState({ modalOpen: true }, '');

    // Sync overlays to the actual rendered image area inside the img element
    // (object-fit:contain means the rendered area may be smaller than offsetWidth/Height)
    // Assigned later (equirect only) to also keep the roll layers' geometry in sync.
    let rollGeometrySync = null;
    const syncOverlays = () => {
        const ew = img.offsetWidth, eh = img.offsetHeight;
        const nw = img.naturalWidth || ew, nh = img.naturalHeight || eh;
        const imgAspect = nw / nh, boxAspect = ew / eh;
        let rw, rh, rl, rt;
        if (imgAspect > boxAspect) {
            rw = ew; rh = ew / imgAspect;
            rl = img.offsetLeft; rt = img.offsetTop + (eh - rh) / 2;
        } else {
            rh = eh; rw = eh * imgAspect;
            rt = img.offsetTop; rl = img.offsetLeft + (ew - rw) / 2;
        }
        [gridOverlay, boundsOverlay, annotationOverlay, maskOverlay].forEach(ov => {
            ov.style.left   = rl + 'px';
            ov.style.top    = rt + 'px';
            ov.style.width  = rw + 'px';
            ov.style.height = rh + 'px';
            ov.style.transformOrigin = (rw / 2) + 'px ' + (rh / 2) + 'px';
        });
        // img transformOrigin is relative to img's own top-left (offsetLeft/Top)
        img.style.transformOrigin = (rl - img.offsetLeft + rw / 2) + 'px ' + (rt - img.offsetTop + rh / 2) + 'px';
        if (rollGeometrySync) rollGeometrySync();
    };
    img.addEventListener('load', () => {
        if (img.naturalWidth && img.naturalHeight) {
            imageWrapper.style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`;
        }
        syncOverlays();
    });
    const ro = new ResizeObserver(syncOverlays);
    ro.observe(imageWrapper);

    makeModalResizable(modalContent, syncOverlays);

    // Pan/Zoom
    let scale = 1, minScale = 1, panX = 0, panY = 0, isPanning = false, startPanX = 0, startPanY = 0, panOriginX = 0, panOriginY = 0;
    // Equirectangular roll state: horizontal offset is applied via CSS background-position
    // (see roll layers below) so it wraps seamlessly instead of clamping like panX.
    let rollOffsetPx = 0, startRollOffset = 0;
    let rollBase = null, rollGrid = null, rollBounds = null, rollMask = null;
    const clamp = (val, min, max) => Math.min(Math.max(val, min), max);
    const updateTransform = () => {
        const transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
        img.style.transform = gridOverlay.style.transform = boundsOverlay.style.transform = annotationOverlay.style.transform = maskOverlay.style.transform = transform;
        if (rollBase) {
            // Roll layers only ever translate vertically (panY) + zoom; the horizontal
            // component is handled by background-position so it can wrap infinitely.
            const rollTransform = `translate(0px, ${panY}px) scale(${scale})`;
            rollBase.style.transform = rollGrid.style.transform = rollBounds.style.transform = rollMask.style.transform = rollTransform;
        }
    };

    // Compute scale needed to fill the fullscreen wrapper.
    // Reads actual rendered img rect (post-CSS, pre-transform) and wrapper rect.
    // Picks the scale that fills the screen on the constrained axis without overflowing the other.
    const computeFillScale = () => {
        const wRect = imageWrapper.getBoundingClientRect();
        const iRect = img.getBoundingClientRect();
        if (!iRect.width || !iRect.height || !wRect.width || !wRect.height) return 1;
        const nw = img.naturalWidth, nh = img.naturalHeight;
        if (!nw || !nh) return 1;
        const imgAspect = nw / nh;
        const wrapperAspect = wRect.width / wRect.height;
        if (imgAspect >= wrapperAspect) {
            // Wide image (equirect): contained by width → letterbox bars → return 1
            return 1;
        }
        // Square/portrait image (fisheye): should be contained by height → pillarbox bars
        // CSS is actually containing by width, so scaleY fills height without width overflow
        return Math.max(1, wRect.height / iRect.height);
    };

    const onWheel = e => {
        e.preventDefault();
        const rect = imageWrapper.getBoundingClientRect();
        const imgNaturalWidth = img.naturalWidth || img.offsetWidth;
        const imgNaturalHeight = img.naturalHeight || img.offsetHeight;
        const wrapperAspect = rect.width / rect.height;
        const imgAspect = imgNaturalWidth / imgNaturalHeight;
        let baseImgWidth, baseImgHeight, baseImgX, baseImgY;
        if (imgAspect > wrapperAspect) {
            baseImgWidth = rect.width;
            baseImgHeight = rect.width / imgAspect;
            baseImgX = 0;
            baseImgY = (rect.height - baseImgHeight) / 2;
        } else {
            baseImgWidth = rect.height * imgAspect;
            baseImgHeight = rect.height;
            baseImgX = (rect.width - baseImgWidth) / 2;
            baseImgY = 0;
        }
        const centerX = baseImgX + baseImgWidth / 2;
        const centerY = baseImgY + baseImgHeight / 2;
        const mouseX = e.clientX - rect.left - centerX;
        const mouseY = e.clientY - rect.top - centerY;
        const newScale = clamp(scale * (e.deltaY > 0 ? 0.9 : 1.1), minScale, 8);
        const newPanX = mouseX - (mouseX - panX) * (newScale / scale);
        const newPanY = mouseY - (mouseY - panY) * (newScale / scale);
        scale = newScale;
        if (scale <= minScale + 0.01) { panX = 0; panY = 0; } else {
            const maxPanX = baseImgWidth * (scale - 1) / 2;
            const maxPanY = baseImgHeight * (scale - 1) / 2;
            panX = clamp(newPanX, -maxPanX, maxPanX);
            panY = clamp(newPanY, -maxPanY, maxPanY);
        }
        updateTransform();
    };
    const onMouseMove = e => { if (!isPanning) return; const rect = imageWrapper.getBoundingClientRect(); const imgNaturalWidth = img.naturalWidth || img.offsetWidth; const imgNaturalHeight = img.naturalHeight || img.offsetHeight; const wrapperAspect = rect.width / rect.height; const imgAspect = imgNaturalWidth / imgNaturalHeight; let baseImgWidth, baseImgHeight; if (imgAspect > wrapperAspect) { baseImgWidth = rect.width; baseImgHeight = rect.width / imgAspect; } else { baseImgWidth = rect.height * imgAspect; baseImgHeight = rect.height; } const maxPanY = baseImgHeight * (scale - 1) / 2; panY = clamp(startPanY + (e.clientY - panOriginY), -maxPanY, maxPanY); if (isEquirect) { rollOffsetPx = startRollOffset + (e.clientX - panOriginX) / scale; if (rollGeometrySync) rollGeometrySync(); } else { const maxPanX = baseImgWidth * (scale - 1) / 2; panX = clamp(startPanX + (e.clientX - panOriginX), -maxPanX, maxPanX); } updateTransform(); };
    const onMouseUp = () => { isPanning = false; imageWrapper.style.cursor = isEquirect ? 'grab' : 'default'; window.removeEventListener('mousemove', onMouseMove); window.removeEventListener('mouseup', onMouseUp); };
    const onMouseDown = e => { if (e.button !== 0) return; e.preventDefault(); isPanning = true; imageWrapper.style.cursor = 'grabbing'; panOriginX = e.clientX; panOriginY = e.clientY; startPanX = panX; startPanY = panY; startRollOffset = rollOffsetPx; window.addEventListener('mousemove', onMouseMove); window.addEventListener('mouseup', onMouseUp); };
    imageWrapper.addEventListener('wheel', onWheel); imageWrapper.addEventListener('mousedown', onMouseDown);

    if (isEquirect) {
        // Roll layers mirror the (now-hidden) img/grid/bounds/mask elements using
        // CSS background-image + repeat-x, which tiles the 360° panorama infinitely.
        // Dragging just moves background-position, so the wrap point never has to be
        // handled explicitly - the browser's own tiling makes it seamless.
        img.style.visibility = 'hidden';
        gridOverlay.style.visibility = 'hidden';
        boundsOverlay.style.visibility = 'hidden';
        maskOverlay.style.visibility = 'hidden';

        rollBase = createEl('div', { style: { position: 'absolute', backgroundRepeat: 'repeat-x', pointerEvents: 'none', zIndex: 1, opacity: '1' } });
        rollGrid = createEl('div', { style: { position: 'absolute', backgroundRepeat: 'repeat-x', pointerEvents: 'none', zIndex: 10, opacity: '0' } });
        rollBounds = createEl('div', { style: { position: 'absolute', backgroundRepeat: 'repeat-x', pointerEvents: 'none', zIndex: 13, opacity: '0' } });
        rollMask = createEl('div', { style: { position: 'absolute', backgroundRepeat: 'repeat-x', pointerEvents: 'none', zIndex: 12, opacity: '0' } });
        imageWrapper.append(rollBase, rollGrid, rollBounds, rollMask);
        imageWrapper.style.cursor = 'grab';

        const applyRollGeometry = () => {
            const ew = img.offsetWidth, eh = img.offsetHeight;
            const nw = img.naturalWidth || ew, nh = img.naturalHeight || eh;
            const imgAspect = nw / nh, boxAspect = ew / eh;
            let rw, rh, rl, rt;
            if (imgAspect > boxAspect) {
                rw = ew; rh = ew / imgAspect;
                rl = img.offsetLeft; rt = img.offsetTop + (eh - rh) / 2;
            } else {
                rh = eh; rw = eh * imgAspect;
                rt = img.offsetTop; rl = img.offsetLeft + (ew - rw) / 2;
            }
            [rollBase, rollGrid, rollBounds, rollMask].forEach(layer => {
                layer.style.left = rl + 'px';
                layer.style.top = rt + 'px';
                layer.style.width = rw + 'px';
                layer.style.height = rh + 'px';
                layer.style.backgroundSize = `${rw}px ${rh}px`;
                layer.style.backgroundPositionX = rollOffsetPx + 'px';
                layer.style.backgroundPositionY = '0';
            });
        };
        rollGeometrySync = applyRollGeometry;

        // Mirror src + opacity from the hidden source <img> onto its roll layer
        // whenever either changes (initial load, enhance filter, checkbox toggle).
        const mirrorLayer = (sourceImg, rollLayer) => {
            const sync = () => {
                rollLayer.style.backgroundImage = sourceImg.src ? `url("${sourceImg.src}")` : 'none';
            };
            new MutationObserver(sync).observe(sourceImg, { attributes: true, attributeFilter: ['src'] });
            sync();
        };
        mirrorLayer(img, rollBase);
        mirrorLayer(gridOverlay, rollGrid);
        mirrorLayer(boundsOverlay, rollBounds);
        mirrorLayer(maskOverlay, rollMask);
        // Opacity toggles (grid/bounds/mask checkboxes) set style.opacity on the hidden
        // source <img>; mirror that onto the roll layer too.
        const mirrorOpacity = (sourceImg, rollLayer) => {
            new MutationObserver(() => { rollLayer.style.opacity = sourceImg.style.opacity || '0'; })
                .observe(sourceImg, { attributes: true, attributeFilter: ['style'] });
        };
        mirrorOpacity(gridOverlay, rollGrid);
        mirrorOpacity(boundsOverlay, rollBounds);
        mirrorOpacity(maskOverlay, rollMask);

        applyRollGeometry();
    }

    // Fullscreen
    fullscreenBtn.addEventListener('click', () => {
        if (imageWrapper.requestFullscreen) imageWrapper.requestFullscreen();
        else if (imageWrapper.webkitRequestFullscreen) imageWrapper.webkitRequestFullscreen();
    });
    const onFullscreenChange = () => {
        if (!document.fullscreenElement) {
            minScale = 1; scale = 1; panX = 0; panY = 0;
            imageWrapper.style.opacity = '';
            img.style.width = '';
            img.style.height = '';
            img.style.objectFit = '';
            img.style.maxWidth = '';
            img.style.maxHeight = '';
            updateTransform();
            syncOverlays();
        } else {
            // Force the image to fill the fullscreen wrapper while preserving aspect ratio.
            img.style.width = '100%';
            img.style.height = '100%';
            img.style.objectFit = 'contain';
            img.style.maxWidth = 'none';
            img.style.maxHeight = 'none';
            scale = 1; minScale = 1; panX = 0; panY = 0;
            updateTransform();
            syncOverlays();
        }
        gridOverlay.style.opacity = gridCheckbox?.checked ? '0.6' : '0';
        boundsOverlay.style.opacity = boundsCheckbox?.checked ? '0.8' : '0';
        annotationOverlay.style.opacity = annotationCheckbox?.checked ? '0.6' : '0';
        maskOverlay.style.opacity = maskCheckbox?.checked ? '1' : '0';
    };
    document.addEventListener('fullscreenchange', onFullscreenChange);

    // Download — composite active overlays onto the image at full resolution
    downloadBtn.addEventListener('click', () => {
        const hasGrid = gridCheckbox?.checked && gridOverlay.src && gridOverlay.complete;
        const hasBounds = boundsCheckbox?.checked && boundsOverlay.src && boundsOverlay.complete;
        const hasAnnotation = annotationCheckbox?.checked && annotationOverlay.src && annotationOverlay.complete;
        const hasMask = maskCheckbox?.checked && maskOverlay.src && maskOverlay.complete;
        if (!hasGrid && !hasBounds && !hasAnnotation && !hasMask) {
            const link = document.createElement('a');
            link.href = imageUrl;
            link.download = title;
            link.click();
            return;
        }
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.filter = img.style.filter || 'none';
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        ctx.filter = 'none';
        if (hasGrid) {
            ctx.globalAlpha = 0.6;
            ctx.drawImage(gridOverlay, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
        }
        if (hasAnnotation) {
            ctx.globalAlpha = 0.6;
            ctx.drawImage(annotationOverlay, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
        }
        if (hasMask) {
            // Mask overlay's own alpha channel already encodes sky
            // (transparent) vs. foreground (opaque black). Drawn before
            // the camera bounds overlay so "Vis kameragrenser" stays
            // visible on top.
            ctx.drawImage(maskOverlay, 0, 0, canvas.width, canvas.height);
        }
        if (hasBounds) {
            ctx.globalAlpha = 0.8;
            ctx.drawImage(boundsOverlay, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
        }
        canvas.toBlob(blob => {
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = title.replace(/\.[^.]+$/, '') + '_overlay.jpg';
            link.click();
            URL.revokeObjectURL(url);
        }, 'image/jpeg', 0.92);
    });

    // Navigation button handlers - use currentMediaList for dynamic bounds checking
    if (prevBtn && nextBtn && mediaList && mediaIndex >= 0) {
        prevBtn.addEventListener('click', () => {
            if (mediaIndex > 0) {
                const prevItem = currentMediaList[mediaIndex - 1];
                const rect = modalContent.getBoundingClientRect();
                lastModalDimensions = { width: rect.width, height: rect.height };
                closeModal(true);
                if (prevItem.isVideo) {
                    showVideoPreview(prevItem.url, prevItem.name, currentMediaList, mediaIndex - 1, lastModalDimensions);
                } else {
                    showImagePreview(prevItem.url, prevItem.name, currentMediaList, mediaIndex - 1, lastModalDimensions);
                }
            }
        });
        nextBtn.addEventListener('click', () => {
            if (mediaIndex < currentMediaList.length - 1) {
                const nextItem = currentMediaList[mediaIndex + 1];
                const rect = modalContent.getBoundingClientRect();
                lastModalDimensions = { width: rect.width, height: rect.height };
                closeModal(true);
                if (nextItem.isVideo) {
                    showVideoPreview(nextItem.url, nextItem.name, currentMediaList, mediaIndex + 1, lastModalDimensions);
                } else {
                    showImagePreview(nextItem.url, nextItem.name, currentMediaList, mediaIndex + 1, lastModalDimensions);
                }
            }
        });
    }

    // Keyboard - check bounds dynamically instead of using disabled state
    modalBackdrop.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            e.preventDefault();
            closeButton.click();
        } else if (e.key === 'ArrowLeft' && e.shiftKey && prevBtn && mediaIndex > 0) {
            e.preventDefault();
            prevBtn.click();
        } else if (e.key === 'ArrowRight' && e.shiftKey && nextBtn && mediaIndex < currentMediaList.length - 1) {
            e.preventDefault();
            nextBtn.click();
        }
    });

    // Close
    const closeModal = (skipHistory = false) => {
        ro.disconnect();
        document.removeEventListener('fullscreenchange', onFullscreenChange);
        imageWrapper.removeEventListener('wheel', onWheel);
        imageWrapper.removeEventListener('mousedown', onMouseDown);
        modalBackdrop.remove();
        if (!skipHistory) history.back();
    };
    closeButton.addEventListener('click', () => closeModal(false));

    modalBackdrop.setAttribute('tabindex', '0');
    setTimeout(() => modalBackdrop.focus(), 100);
}

/**
 * Renders the results of a completed download task in the results panel.
 * @param {object} resultData - The data object from the backend, containing files and errors.
 * @param {object} dom - The DOM element cache.
 * @param {boolean} hevcSupported - Whether the user's browser supports HEVC.
 */
export function displayResults(resultData, dom, hevcSupported, stationsData = null) {
    // Store stations data for preview modals
    if (stationsData) {
        previewStationsData = stationsData;
    }

    dom.resultsLog.innerHTML = '';
    const stationResults = resultData.files || {};
    
    // Build flat list of all media items for navigation - stored globally for dynamic updates
    currentMediaList = [];
    if (Object.keys(stationResults).length > 0) {
        Object.keys(stationResults).sort().forEach((stationCode) => {
            const timeGroupedFiles = stationResults[stationCode];
            const startHour = parseInt(dom.hourSelect.value, 10);
            
            const getSortKey = (key) => {
                const isRange = key.includes(' - ');
                const timePart = isRange ? key.split(' - ')[1] : key;
                const hour = parseInt(timePart.split(':')[0], 10);
                const normalizedHour = hour < startHour ? hour + 24 : hour;
                return `${String(normalizedHour).padStart(2, '0')}:${timePart.split(':')[1]}:${isRange ? '1' : '0'}`;
            };
            
            Object.keys(timeGroupedFiles).sort((a, b) => getSortKey(a).localeCompare(getSortKey(b))).forEach((time) => {
                timeGroupedFiles[time].forEach(file => {
                    const isVideo = file.url.endsWith('.mp4');
                    currentMediaList.push({
                        url: file.url,
                        name: file.name,
                        thumb_url: file.thumb_url,
                        isVideo: isVideo,
                        duration: file.duration,
                        start_time: file.start_time,
                        alternatives: file.alternatives || []
                    });
                });
            });
        });
    }
    
    if (Object.keys(stationResults).length > 0) {
        dom.resultsLog.appendChild(createEl('h4', { textContent: t('downloaded_files_title') }));
        Object.keys(stationResults).sort().forEach((stationCode, stationIndex) => {
            if (stationIndex > 0) dom.resultsLog.appendChild(createEl('hr', { className: 'station-separator' }));
            dom.resultsLog.appendChild(createEl('h5', { textContent: t('station_results_title', { station_code: stationCode }) }));
            const timeGroupedFiles = stationResults[stationCode];
            const startHour = parseInt(dom.hourSelect.value, 10);
            
            const getSortKey = (key) => {
                const isRange = key.includes(' - ');
                const timePart = isRange ? key.split(' - ')[1] : key;
                const hour = parseInt(timePart.split(':')[0], 10);
                const normalizedHour = hour < startHour ? hour + 24 : hour;
                return `${String(normalizedHour).padStart(2, '0')}:${timePart.split(':')[1]}:${isRange ? '1' : '0'}`;
            };
            Object.keys(timeGroupedFiles).sort((a, b) => getSortKey(a).localeCompare(getSortKey(b))).forEach((time, timeIndex) => {
               const timeSetDiv = createEl('div', { className: `time-set ${timeIndex % 2 === 0 ? 'time-set-even' : 'time-set-odd'}` });
                timeSetDiv.appendChild(createEl('h6', { textContent: t('time_results_title', { time: time }) }));
                const ul = createEl('ul', { className: 'result-list' });
                timeGroupedFiles[time].forEach(file => {
                    // Find index in flat media list for navigation
                    const mediaIndex = currentMediaList.findIndex(m => m.url === file.url && m.name === file.name);
                    const li = createEl('li');
                    const isVideo = file.url.endsWith('.mp4');

                    const getShortName = (filename) => {
                        if (filename.includes('_image_long_stacked.jpg')) return 'bhL';
                        if (filename.includes('_image_lowres_long_stacked.jpg')) return 'blL';
                        if (filename.includes('_hires_fisheye.jpg')) return 'fe';
                        if (filename.includes('_lowres_fisheye.jpg')) return 'fe';
                        if (filename.includes('_hires_equirect.jpg')) return 'eq';
                        if (filename.includes('_lowres_equirect.jpg')) return 'eq';
                        if (filename.endsWith('_teq.mp4')) return 'teq';
                        if (filename.endsWith('_tfe.mp4')) return 'tfe';
                        if (filename.endsWith('_teqh.mp4')) return 'teqh';
                        if (filename.endsWith('_tfeh.mp4')) return 'tfeh';
                        const durVideoMatch = filename.match(/_dur(\d+)_(hires|lowres)\.mp4$/);
                        if (durVideoMatch) return (durVideoMatch[2] === 'hires' ? 'vh' : 'vl') + durVideoMatch[1];
                        const durImageMatch = filename.match(/_dur(\d+)_(image_long|image_lowres_long)\.jpg$/);
                        if (durImageMatch) return (durImageMatch[2] === 'image_long' ? 'bhl' : 'bll') + durImageMatch[1];
                        const typeMap = { '_hires_hevc.mp4': 'vh', '_lowres_hevc.mp4': 'vl', '_hires.mp4': 'vh', '_lowres.mp4': 'vl', '_image_long.jpg': 'bhl', '_image_lowres_long.jpg': 'bll', '_image.jpg': 'bh', '_image_lowres.jpg': 'blr' };
                        let baseType = filename;
                        let isOverlay = false;
                        if (baseType.includes('_flight_overlay')) { isOverlay = true; baseType = baseType.replace('_flight_overlay', ''); }
                        else if (baseType.includes('_overlay')) { isOverlay = true; baseType = baseType.replace('_overlay', ''); }
                        for (const key in typeMap) {
                            if (baseType.endsWith(key)) return typeMap[key] + (isOverlay ? 's' : '');
                        }
                        return filename;
                    };

                    if (file.thumb_url) {
                        const thumbContainer = createEl('div', { className: `thumbnail-container${isVideo ? ' video' : ''}` });
                        thumbContainer.appendChild(createEl('img', { src: file.thumb_url, alt: file.name, className: 'thumbnail-preview' }));

                        // Both video and image thumbnails open a preview player with navigation
                        thumbContainer.style.cursor = 'pointer';
                        thumbContainer.addEventListener('click', () => {
                            if (isVideo) {
                                showVideoPreview(file.url, file.name, currentMediaList, mediaIndex);
                            } else {
                                showImagePreview(file.url, file.name, currentMediaList, mediaIndex);
                            }
                        });
                        li.appendChild(thumbContainer);
                     } else {
                        const fallbackShort = getShortName(file.name);
                        const fallbackLink = createEl('a', { href: '#', textContent: fallbackShort, title: file.name });
                        fallbackLink.addEventListener('click', (e) => {
                            e.preventDefault();
                            if (file.url.endsWith('.mp4')) {
                                showVideoPreview(file.url, file.name, currentMediaList, mediaIndex);
                            } else {
                                showImagePreview(file.url, file.name, currentMediaList, mediaIndex);
                            }
                        });
                        li.appendChild(fallbackLink);
                    }

               
                     const linksContainer = createEl('div', { className: 'alternate-links' });
                    const allFilesForThisThumb = [{ url: file.url, name: file.name }, ...(file.alternatives || [])];
                    const preferredLinks = {};

                    allFilesForThisThumb.forEach(f => {
                        const shortName = getShortName(f.name);
                        const isHevc = f.name.includes('_hevc.mp4');
                        const existing = preferredLinks[shortName];
                
 
                        if (!existing) {
                            preferredLinks[shortName] = f;
                       
                    
                         
                        } else {
                            const existingIsHevc = existing.name.includes('_hevc.mp4');
                            // Prefer the H.264 (non-HEVC) copy so video scrubbing works reliably.
                            if (!isHevc && existingIsHevc) preferredLinks[shortName] = f;
                        }
                 
                   });
                    Object.entries(preferredLinks).sort((a, b) => a[0].localeCompare(b[0])).forEach(([shortName, linkInfo]) => {
                        const linkEl = createEl('a', { href: '#', textContent: shortName });
                        linkEl.addEventListener('click', (e) => {
                            e.preventDefault();
                            const linkIndex = currentMediaList.findIndex(m => m.url === linkInfo.url && m.name === linkInfo.name);
                            if (linkInfo.url.endsWith('.mp4')) {
                                let previewList = currentMediaList;
                                let previewIndex = linkIndex >= 0 ? linkIndex : mediaIndex;
                                // If the video alternative is not a top-level media item, splice its
                                // duration/start_time into the current slot so the scrubber works.
                                if (linkIndex < 0 && (linkInfo.duration != null || linkInfo.start_time != null) && mediaIndex >= 0) {
                                    previewList = currentMediaList.map((m, i) => i === mediaIndex
                                        ? { ...m, url: linkInfo.url, name: linkInfo.name, isVideo: true, duration: linkInfo.duration, start_time: linkInfo.start_time }
                                        : m);
                                }
                                showVideoPreview(linkInfo.url, linkInfo.name, previewList, previewIndex);
                            } else {
                                showImagePreview(linkInfo.url, linkInfo.name, currentMediaList, linkIndex >= 0 ? linkIndex : mediaIndex);
                            }
                        });
                        linksContainer.appendChild(linkEl);
                    });
                    if (linksContainer.hasChildNodes()) li.appendChild(linksContainer);
                    ul.appendChild(li);
                });
                timeSetDiv.appendChild(ul);
                dom.resultsLog.appendChild(timeSetDiv);
            });
        });
    }
    
    const errorData = resultData.errors || {};
    if (Object.keys(errorData).length > 0) {
        dom.resultsLog.appendChild(createEl('h4', { textContent: t('error_messages_title') }));
        Object.entries(errorData).forEach(([stationCode, errors]) => {
            dom.resultsLog.appendChild(createEl('h5', { textContent: t('station_results_title', { station_code: stationCode }) }));
            const errorUl = createEl('ul');
            errors.forEach(error => errorUl.appendChild(createEl('li', { className: 'error-msg', textContent: translateMessage(error) })));
            dom.resultsLog.appendChild(errorUl);
        });
    }
    
    if (dom.resultsLog.innerHTML === '' && resultData.status === 'complete') {
        dom.resultsLog.appendChild(createEl('h4', { textContent: t('no_files_found') }));
    }
}

/**
 * Creates and displays the modal window for viewing a live video stream.
 * @param {string} stationId
 * @param {number} cameraNum
 * @param {string} resolution
 * @param {string} streamTaskId
 */
export function showVideoModal(stationId, cameraNum, resolution, streamTaskId, onRetry, stationsData) {
    if (activeStreamTaskId) {
        hideVideoModal();
    }
    activeStreamTaskId = streamTaskId;

    const modalBackdrop = createEl('div', { id: 'video-modal-backdrop' });
    const modalContent = createEl('div', { id: 'video-modal-content' });
    const stationInfo = stationsData?.[stationId]?.station;
    const astronomy = stationsData?.[stationId]?.astronomy;
    const displayName = stationInfo?.display_name || (stationInfo?.name ? stationInfo.name.charAt(0).toUpperCase() + stationInfo.name.slice(1) : stationId);

    // Calculate sun altitude on the fly
    let sunAltText = '';
    if (astronomy && astronomy.latitude && astronomy.longitude) {
        const sunAlt = getSunAltitude(new Date(), astronomy.latitude, astronomy.longitude);
        sunAltText = ` | ${t('sun_altitude')}: ${sunAlt.toFixed(1)}°`;
    }

    // Build title with coordinates, elevation, and sun altitude
    let titleText = `${displayName} – ${cameraNum}`;
    if (astronomy) {
        const lat = `${astronomy.latitude.toFixed(3)}N`;
        const lon = `${astronomy.longitude.toFixed(3)}E`;
        const elev = astronomy.elevation ? `${astronomy.elevation}m` : '';
        titleText += ` (${lat}, ${lon}${elev ? `, ${elev}` : ''}${sunAltText})`;
    }
    const header = createEl('div', { className: 'preview-header' });
    const modalTitle = createEl('h3', { id: 'video-modal-title', textContent: titleText, className: 'preview-title' });
    const closeButton = createEl('button', { className: 'preview-close-btn', textContent: '×' });
    closeButton.addEventListener('click', hideVideoModal);
    header.append(modalTitle, closeButton);

    const videoContainer = createEl('div', { id: 'video-container', style: { aspectRatio: resolution === 'lowres' ? '800 / 448' : '1920 / 1080' } });
    const videoEl = createEl('video', { id: 'live-video', muted: true, autoplay: true, playsinline: true });
    const gridOverlay = createEl('img', { id: 'grid-overlay-image' });
    const annotationOverlay = createEl('img', { id: 'annotation-overlay-image' });
    // Explicitly set opacity 0 to hide initially
    gridOverlay.style.opacity = '0';
    annotationOverlay.style.opacity = '0';
    const statusEl = createEl('p', { id: 'video-status', textContent: t('modal_starting_stream') });
    const controlsContainer = createEl('div', { className: 'video-controls-container' });
    const gridToggleContainer = createEl('div', { id: 'grid-toggle-container', style: 'display: none;' });
    const gridCheckbox = createEl('input', { type: 'checkbox', id: 'grid-overlay-toggle' });
    const gridLabel = createEl('label', { textContent: t('modal_grid_toggle'), htmlFor: 'grid-overlay-toggle' });
    gridToggleContainer.append(gridCheckbox, gridLabel);
    const annotationToggleContainer = createEl('div', { id: 'annotation-toggle-container', style: 'display: none;' });
    const annotationCheckbox = createEl('input', { type: 'checkbox', id: 'annotation-overlay-toggle' });
    const annotationLabel = createEl('label', { textContent: t('modal_annotation_toggle'), htmlFor: 'annotation-overlay-toggle' });
    annotationToggleContainer.append(annotationCheckbox, annotationLabel);
    const fullscreenButton = createEl('button', { id: 'fullscreen-btn', textContent: t('modal_fullscreen_button') });
    controlsContainer.append(gridToggleContainer, annotationToggleContainer, fullscreenButton);

    // Filter controls
    const liveFilterControls = createEl('div', { className: 'preview-filter-controls' });
    const liveBrightnessSlider = createEl('input', { type: 'range', min: '0.5', max: '2', step: '0.1', value: '1', className: 'preview-slider', id: 'live-brightness-slider' });
    const liveContrastSlider = createEl('input', { type: 'range', min: '0.5', max: '2', step: '0.1', value: '1', className: 'preview-slider', id: 'live-contrast-slider' });
    const liveSaturationSlider = createEl('input', { type: 'range', min: '0', max: '3', step: '0.1', value: '1', className: 'preview-slider', id: 'live-saturation-slider' });
    const liveResetBtn = createEl('button', { className: 'preview-control-btn reset', textContent: t('reset_filters'), title: t('reset_filters') });
    const liveBrightnessWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    liveBrightnessWrapper.append(createEl('label', { textContent: t('brightness'), htmlFor: 'live-brightness-slider', className: 'preview-filter-label' }), liveBrightnessSlider);
    const liveContrastWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    liveContrastWrapper.append(createEl('label', { textContent: t('contrast'), htmlFor: 'live-contrast-slider', className: 'preview-filter-label' }), liveContrastSlider);
    const liveSaturationWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    liveSaturationWrapper.append(createEl('label', { textContent: t('saturation'), htmlFor: 'live-saturation-slider', className: 'preview-filter-label' }), liveSaturationSlider);
    liveFilterControls.append(liveResetBtn, liveBrightnessWrapper, liveContrastWrapper, liveSaturationWrapper);

    let baseStatusText = t('modal_starting_stream');
    let countdownSuffix = '';
    let bitrateText = '';

    const renderStatusLine = () => {
        if (!statusEl) return;
        const parts = [baseStatusText];
        if (bitrateText) parts.push(bitrateText);
        if (countdownSuffix) parts.push(countdownSuffix);
        statusEl.textContent = parts.join(' | ');
    };

    const setBaseStatusText = (text) => {
        baseStatusText = text;
        renderStatusLine();
    };

    const setBitrateText = (text) => {
        bitrateText = text;
        renderStatusLine();
    };

    videoContainer.append(videoEl, gridOverlay, annotationOverlay);
    modalContent.append(header, statusEl, videoContainer, controlsContainer, liveFilterControls);
    modalBackdrop.appendChild(modalContent);
    document.body.appendChild(modalBackdrop);

    // Push history state for back button handling
    history.pushState({ modalOpen: true }, '');
    
    // --- Overlay Sizing to Match Video Display Area ---
    const updateOverlaySizing = () => {
        if (!videoEl.videoWidth || !videoEl.videoHeight) return;
        
        const containerRect = videoContainer.getBoundingClientRect();
        const videoAspect = videoEl.videoWidth / videoEl.videoHeight;
        const containerAspect = containerRect.width / containerRect.height;
        
        let displayWidth = containerRect.width;
        let displayHeight = containerRect.height;
        let offsetX = 0;
        let offsetY = 0;
        
        // Calculate actual video display area (letterboxed)
        if (videoAspect > containerAspect) {
            // Video is wider than container - black bars on top/bottom
            displayHeight = containerRect.width / videoAspect;
            offsetY = (containerRect.height - displayHeight) / 2;
        } else if (videoAspect < containerAspect) {
            // Video is taller than container - black bars on sides
            displayWidth = containerRect.height * videoAspect;
            offsetX = (containerRect.width - displayWidth) / 2;
        }
        
        // Apply to overlays
        const overlays = [gridOverlay, annotationOverlay];
        overlays.forEach(overlay => {
            overlay.style.width = `${displayWidth}px`;
            overlay.style.height = `${displayHeight}px`;
            overlay.style.left = `${offsetX}px`;
            overlay.style.top = `${offsetY}px`;
            // Set transform-origin to negative of offset so zoom happens from container's (0,0)
            // This makes overlays zoom from the same point as the video
            overlay.style.transformOrigin = `${-offsetX}px ${-offsetY}px`;
        });
    };
    
    videoEl.addEventListener('loadedmetadata', () => {
        if (videoEl.videoHeight > 0) {
            videoContainer.style.aspectRatio = videoEl.videoWidth / videoEl.videoHeight;
            updateOverlaySizing();
        }
    });
    window.addEventListener('resize', updateOverlaySizing);

    makeModalResizable(modalContent, updateOverlaySizing);

    gridCheckbox.addEventListener('change', () => { gridOverlay.style.opacity = gridCheckbox.checked ? '0.6' : '0'; });
    annotationCheckbox.addEventListener('change', () => { annotationOverlay.style.opacity = annotationCheckbox.checked ? '0.6' : '0'; });
    
    // --- Pan/Zoom Logic (kept condensed) ---
    let scale=1, panX=0, panY=0, isPanning=false, startPanX=0, startPanY=0, panOriginX=0, panOriginY=0;
    const clamp = (val, min, max) => Math.min(Math.max(val, min), max);
    const updateTransform = () => { 
        const transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
        videoEl.style.transform = gridOverlay.style.transform = annotationOverlay.style.transform = transform;
    };
    videoEl.style.transformOrigin = gridOverlay.style.transformOrigin = annotationOverlay.style.transformOrigin = '0 0';
    const onWheel = e => { e.preventDefault(); const rect=videoContainer.getBoundingClientRect(); const videoRect=videoEl.getBoundingClientRect(); const mouseX=e.clientX-rect.left; const mouseY=e.clientY-rect.top; const newScale=clamp(scale*(e.deltaY>0?0.9:1.1),1,8); const newPanX=mouseX-(mouseX-panX)*(newScale/scale); const newPanY=mouseY-(mouseY-panY)*(newScale/scale); scale=newScale; if(scale<=1.01){panX=0;panY=0;}else{panX=clamp(newPanX,-(videoRect.width*(scale-1)),0); panY=clamp(newPanY,-(videoRect.height*(scale-1)),0);} updateTransform(); };
    const onMouseMove = e => { if(!isPanning)return; const videoRect=videoEl.getBoundingClientRect(); panX=clamp(startPanX+(e.clientX-panOriginX),-(videoRect.width*(scale-1)),0); panY=clamp(startPanY+(e.clientY-panOriginY),-(videoRect.height*(scale-1)),0); updateTransform(); };
    const onMouseUp = () => { isPanning=false; videoContainer.style.cursor='grab'; window.removeEventListener('mousemove',onMouseMove); window.removeEventListener('mouseup',onMouseUp); };
    const onMouseDown = e => { if(e.button!==0)return; e.preventDefault(); isPanning=true; videoContainer.style.cursor='grabbing'; panOriginX=e.clientX; panOriginY=e.clientY; startPanX=panX; startPanY=panY; window.addEventListener('mousemove',onMouseMove); window.addEventListener('mouseup',onMouseUp); };
    let cursorIdleTimer=null; const handleIdleCursor=()=>{if(!videoContainer||!document.fullscreenElement)return; videoContainer.style.cursor='default'; clearTimeout(cursorIdleTimer); cursorIdleTimer=setTimeout(()=>{videoContainer.style.cursor='none';},2000);};
    onFullscreenChange = () => { 
        const isFullscreen=!!document.fullscreenElement; 
        fullscreenButton.textContent=isFullscreen?t('modal_exit_fullscreen_button'):t('modal_fullscreen_button'); 
        if(isFullscreen){
            // Delay reset to allow fullscreen styles to apply first
            setTimeout(() => {
                scale=1;panX=0;panY=0;updateTransform();
                updateOverlaySizing();
            }, 100);
            videoContainer.addEventListener('mousemove',handleIdleCursor);
            handleIdleCursor();
        } else {
            videoContainer.removeEventListener('mousemove',handleIdleCursor);
            clearTimeout(cursorIdleTimer);
            videoContainer.style.cursor='grab';
            // Reset transforms when exiting fullscreen
            scale=1;panX=0;panY=0;updateTransform();
            setTimeout(updateOverlaySizing, 100);
        } 
    };
    fullscreenButton.addEventListener('click', () => { if (!document.fullscreenElement) videoContainer.requestFullscreen().catch(err => alert(t('modal_fullscreen_error', { error: err.message }))); else document.exitFullscreen(); });
    videoContainer.addEventListener('wheel', onWheel); videoContainer.addEventListener('mousedown', onMouseDown); document.addEventListener('fullscreenchange', onFullscreenChange);

    // --- Error Handler Wrapper ---
    const triggerRetry = async (reason) => {
        console.warn(`Video playback error (${reason}). Attempting hot-swap to H.264...`);
        
        // Only act if this is still the active stream
        if (activeStreamTaskId === streamTaskId) {
            // 1. Notify user
            if (statusEl) statusEl.textContent = t('modal_status_optimizing') || "Optimizing stream compatibility...";
            
            // 2. Tell backend to switch modes (reusing the tunnel)
            try {
                await api.requestTranscode(streamTaskId);
            } catch (e) {
                console.error("Failed to request transcode:", e);
                // Fallback to full restart if the hot-swap API fails
                if (onRetry) { hideVideoModal(); onRetry(); }
                return;
            }

            // 3. Wait a moment for backend FFmpeg to restart (1.5s is usually enough)
            setTimeout(() => {
                const playlistUrl = `streams/${data.station_id}_${cameraNum}_${data.resolution}/playlist.m3u8`;
                
                if (hls) {
                    hls.stopLoad();
                    // hls.recoverMediaError() might not be enough if codec changed completely
                    hls.loadSource(playlistUrl);
                    hls.startLoad();
                    hls.attachMedia(videoEl);
                } else if (videoEl) {
                    // For native players (Safari), force a reload
                    const currentSrc = videoEl.src;
                    videoEl.src = ''; 
                    videoEl.src = currentSrc; 
                    videoEl.play().catch(e => console.error("Retry play failed:", e));
                }
            }, 1500);
        }
    };

    streamStatusPoller = api.pollStreamStatus(streamTaskId, {
        onStatusUpdate: (data) => {
            if (statusEl) statusEl.textContent = translateMessage(data.message) || t('modal_status_updating');
        },
        onReady: (data) => {
            setBaseStatusText(t('modal_waiting_for_video'));
            const playlistUrl = `streams/${data.station_id}_${cameraNum}_${data.resolution}/playlist.m3u8`;

            const formatCodec = (c) => {
                const s = String(c || '').toLowerCase();
                if (!s) return '';
                if (s === 'h264' || s === 'avc') return 'H.264';
                if (s === 'hevc' || s === 'h265') return 'HEVC';
                return String(c).toUpperCase();
            };

            const codecIndicator = (() => {
                if (!data) return '';
                const inCodec = formatCodec(data.input_codec);
                const outCodec = formatCodec(data.output_codec);
                if (data.transcoding) {
                    if (!inCodec || !outCodec) return '';
                    return `${inCodec} -> ${outCodec}`;
                }
                // Not transcoding: show the active codec only.
                return outCodec || inCodec;
            })();

            let firstFrameSeen = false;
            const markFirstFrameSeen = () => {
                if (firstFrameSeen) return;
                firstFrameSeen = true;
                const liveLabel = t('modal_stream_live');
                setBaseStatusText(codecIndicator ? `${liveLabel} | ${codecIndicator}` : liveLabel);
            };

            const waitForFirstFrame = () => {
                if (!videoEl || firstFrameSeen) return;
                if (typeof videoEl.requestVideoFrameCallback === 'function') {
                    try {
                        videoEl.requestVideoFrameCallback(() => markFirstFrameSeen());
                        return;
                    } catch (e) {
                        // fall through to event-based detection
                    }
                }

                const onTimeUpdate = () => {
                    // When playback has advanced, we have a decoded frame.
                    if (videoEl.currentTime > 0) {
                        videoEl.removeEventListener('timeupdate', onTimeUpdate);
                        markFirstFrameSeen();
                    }
                };
                videoEl.addEventListener('timeupdate', onTimeUpdate);
            };

            // If playback stalls (often due to waiting for a keyframe), keep the user informed.
            videoEl.addEventListener('waiting', () => {
                if (!firstFrameSeen) setBaseStatusText(t('modal_waiting_for_video'));
            }, { once: true });

            if (Hls.isSupported()) {
                hls = new Hls({ maxBufferLength: 2, maxMaxBufferLength: 4, highBufferWatchdogPeriod: 2 });

                let fragBitrateEwmaBps = null;
                const ewmaAlpha = 0.2;
                const fragSizeCache = new Map();
                let lastHeadAtMs = 0;
                const updateBitrateDisplay = () => {
                    const levelIndex = hls?.currentLevel;
                    const levels = hls?.levels;
                    const level = (typeof levelIndex === 'number' && levelIndex >= 0)
                        ? levels?.[levelIndex]
                        : (Array.isArray(levels) && levels.length > 0 ? levels[0] : null);
                    const targetBps = level?.bitrate;

                    const parts = [];
                    if (typeof targetBps === 'number' && isFinite(targetBps) && targetBps > 0) {
                        parts.push(`${Math.round(targetBps / 1000)} kbps`);
                    }
                    if (typeof fragBitrateEwmaBps === 'number' && isFinite(fragBitrateEwmaBps) && fragBitrateEwmaBps > 0) {
                        parts.push(`${Math.round(fragBitrateEwmaBps / 1000)} kbps`);
                    }

                    setBitrateText(parts.join(' / '));
                };

                const updateFromFrag = async (frag, stats) => {
                    try {
                        if (!frag) return;
                        const dur = frag.duration
                            ?? frag._duration
                            ?? ((typeof frag.endPTS === 'number' && typeof frag.startPTS === 'number') ? (frag.endPTS - frag.startPTS) : null);
                        if (!dur || dur <= 0) return;

                        let bytes = stats?.loaded ?? stats?.total;
                        if (!bytes || !isFinite(bytes) || bytes <= 0) {
                            const fragUrl = frag.url
                                ?? frag._url
                                ?? ((frag.baseurl && frag.relurl) ? (frag.baseurl + frag.relurl) : null);
                            if (!fragUrl) return;

                            if (fragSizeCache.has(fragUrl)) {
                                bytes = fragSizeCache.get(fragUrl);
                            } else {
                                const now = Date.now();
                                // Avoid spamming HEAD requests: max ~1 per second.
                                if (now - lastHeadAtMs < 1000) return;
                                lastHeadAtMs = now;

                                // Try HEAD first.
                                let resp = await fetch(fragUrl, { method: 'HEAD', cache: 'no-store' });
                                let len = resp.headers.get('content-length');
                                let parsed = len ? parseInt(len, 10) : NaN;

                                // Some servers omit Content-Length on HEAD; try a 1-byte range GET.
                                if (!isFinite(parsed) || parsed <= 0) {
                                    resp = await fetch(fragUrl, { method: 'GET', headers: { Range: 'bytes=0-0' }, cache: 'no-store' });
                                    const contentRange = resp.headers.get('content-range');
                                    // Format: bytes 0-0/12345
                                    if (contentRange && contentRange.includes('/')) {
                                        const totalStr = contentRange.split('/').pop();
                                        parsed = totalStr ? parseInt(totalStr, 10) : NaN;
                                    } else {
                                        len = resp.headers.get('content-length');
                                        parsed = len ? parseInt(len, 10) : NaN;
                                    }
                                }

                                if (!isFinite(parsed) || parsed <= 0) return;
                                bytes = parsed;
                                fragSizeCache.set(fragUrl, bytes);
                            }
                        }

                        const bps = (bytes * 8) / dur;
                        fragBitrateEwmaBps = (fragBitrateEwmaBps == null) ? bps : (ewmaAlpha * bps + (1 - ewmaAlpha) * fragBitrateEwmaBps);
                        updateBitrateDisplay();
                    } catch (e) {
                        // ignore
                    }
                };

                // Show a placeholder quickly so the user knows bitrate is being determined.
                setBitrateText('bitrate …');
                
                // Catch Fatal HLS Errors (Codec mismatch often triggers MEDIA_ERROR or BUFFER_APPEND_ERROR)
                hls.on(Hls.Events.ERROR, (event, data) => {
                    if (data.fatal) {
                        console.error("HLS Fatal Error:", data);
                        triggerRetry(data.type);
                    }
                });

                // Show configured/selected stream bitrate (level bitrate), and estimate actual stream bitrate
                // from fragment payload size and fragment duration.
                hls.on(Hls.Events.LEVEL_SWITCHED, updateBitrateDisplay);
                hls.on(Hls.Events.LEVEL_LOADED, updateBitrateDisplay);
                hls.on(Hls.Events.MANIFEST_PARSED, updateBitrateDisplay);
                hls.on(Hls.Events.FRAG_LOADED, (event, fragData) => {
                    updateFromFrag(fragData?.frag, fragData?.stats);
                });

                // Fallback: if FRAG_LOADED doesn't provide usable stats, still update based on the currently playing fragment.
                hls.on(Hls.Events.FRAG_CHANGED, (event, fragData) => {
                    updateFromFrag(fragData?.frag, fragData?.stats);
                });

                // Periodic fallback: some Hls.js builds/configs don't populate stats on events.
                // Try to sample the current fragment from internal controllers.
                if (bitrateUpdateInterval) clearInterval(bitrateUpdateInterval);
                bitrateUpdateInterval = setInterval(() => {
                    try {
                        const currentFrag = hls?.streamController?.fragCurrent;
                        if (currentFrag) updateFromFrag(currentFrag, null);
                    } catch (e) {
                        // ignore
                    }
                }, 2000);

                hls.loadSource(playlistUrl);
                hls.attachMedia(videoEl);
                hls.on(Hls.Events.MANIFEST_PARSED, () => {
                    // Catch AbortError specifically from the play promise
                    videoEl.play().catch(e => {
                        console.error("Play Promise Error:", e);
                        triggerRetry(e.name);
                    });

                    waitForFirstFrame();
                });
            } else if (videoEl.canPlayType('application/vnd.apple.mpegurl')) {
                // Native Safari/iOS support
                videoEl.src = playlistUrl;
                videoEl.addEventListener('error', (e) => {
                    console.error("Native Video Error:", videoEl.error);
                    triggerRetry("NativeVideoError");
                });
                videoEl.addEventListener('canplay', () => {
                    videoEl.play().catch(e => triggerRetry(e.name));
                    waitForFirstFrame();
                });
            }

            const timeoutSeconds = data.timeout_seconds || 300;
            let timeLeft = timeoutSeconds;
            stopStreamTimeout = setTimeout(hideVideoModal, timeLeft * 1000);
            streamCountdownInterval = setInterval(() => {
                timeLeft--;
                const minutes = Math.floor(timeLeft / 60);
                const seconds = timeLeft % 60;
                countdownSuffix = t('modal_stream_stops_in', { minutes: minutes, seconds: String(seconds).padStart(2, '0') });
                renderStatusLine();
                if (timeLeft <= 0) clearInterval(streamCountdownInterval);
            }, 1000);
        },
        onError: (data) => {
            if (statusEl) statusEl.textContent = t('modal_status_error', { message: translateMessage(data.message) });
            stopStreamTimeout = setTimeout(hideVideoModal, 5000);
        }
    });
    
    api.fetchStreamGrid(streamTaskId, stationId, cameraNum)
        .then(gridData => {
            if (gridData.success && gridData.grid_url) {
                gridOverlay.src = gridData.grid_url;
                gridToggleContainer.style.display = 'flex';
                annotationToggleContainer.style.display = 'flex';
            }
        })
        .catch(err => console.error("Could not fetch grid overlay:", err));

    const refreshAnnotation = () => {
        api.fetchAnnotation(streamTaskId, stationId, cameraNum)
            .then(annData => {
                if (annData.success && annData.annotation_url) {
                    annotationOverlay.src = annData.annotation_url;
                }
            })
            .catch(err => console.error("Could not fetch annotation overlay:", err));
    };
    refreshAnnotation();
    const annotationRefreshInterval = setInterval(refreshAnnotation, 15000);
    // Store interval so hideVideoModal can clear it
    if (!window._annotationRefreshInterval) window._annotationRefreshInterval = null;
    window._annotationRefreshInterval = annotationRefreshInterval;

    // Live filter handlers
    function updateLiveFilters() {
        videoEl.style.filter = `brightness(${liveBrightnessSlider.value}) contrast(${liveContrastSlider.value}) saturate(${liveSaturationSlider.value})`;
    }
    liveBrightnessSlider.addEventListener('input', updateLiveFilters);
    liveContrastSlider.addEventListener('input', updateLiveFilters);
    liveSaturationSlider.addEventListener('input', updateLiveFilters);
    liveResetBtn.addEventListener('click', () => {
        liveBrightnessSlider.value = 1;
        liveContrastSlider.value = 1;
        liveSaturationSlider.value = 1;
        updateLiveFilters();
    });
}

/**
 * Hides the video modal and cleans up all associated resources.
 */
function hideVideoModal() {
    if (streamCountdownInterval) clearInterval(streamCountdownInterval);
    if (streamStatusPoller) clearInterval(streamStatusPoller);
    if (stopStreamTimeout) clearTimeout(stopStreamTimeout);
    if (bitrateUpdateInterval) clearInterval(bitrateUpdateInterval);
    bitrateUpdateInterval = null;
    if (window._annotationRefreshInterval) { clearInterval(window._annotationRefreshInterval); window._annotationRefreshInterval = null; }
    if (hls) {
        hls.stopLoad();
        hls.destroy();
        hls = null;
    }

    if (activeStreamTaskId) {
        api.stopStream(activeStreamTaskId);
    }

    if (onFullscreenChange) {
        document.removeEventListener('fullscreenchange', onFullscreenChange);
        onFullscreenChange = null;
    }

    document.getElementById('video-modal-backdrop')?.remove();
    activeStreamTaskId = null;
}

// --- Helper Functions used by other modules ---

/**
 * Determines which cameras at a given station would have a lightning strike in their field of view.
 * @param {object} station - The station data object.
 * @param {object} strike - The lightning strike data object.
 * @param {object} cameraFovs - The main camera field of view data.
 * @returns {Array<string>} An array of camera numbers (as strings) that would see the strike.
 */
export function getCamerasInView(station, strike, cameraFovs) {
    const stationFovs = cameraFovs[station.station.id];
    const inViewCams = [];
    if (stationFovs) {
        const bearing = calculateBearing(station.astronomy.latitude, station.astronomy.longitude, strike.lat, strike.lon);
        for (const camName in stationFovs) {
            const camNum = camName.replace('cam', '');
            const fov = stationFovs[camName];
            const halfFov = fov.hFov / 2;
            let lowerBound = fov.centerAzimuth - halfFov,
                upperBound = fov.centerAzimuth + halfFov,
                inFov = false;
            if (lowerBound < 0) {
                inFov = (bearing >= lowerBound + 360 && bearing <= 360) || (bearing >= 0 && bearing <= upperBound);
            } else if (upperBound > 360) {
                inFov = (bearing >= lowerBound && bearing <= 360) || (bearing >= 0 && bearing <= upperBound - 360);
            } else {
                inFov = bearing >= lowerBound && bearing <= upperBound;
            }
            if (inFov) inViewCams.push(camNum);
        }
    }
    return inViewCams;
}

/**
 * Updates the download form based on the selected camera views of a satellite/aircraft pass.
 * @param {object} dom - The DOM element cache.
 * @param {Set<string>} selectedStations - The set of currently selected station IDs.
 * @param {string} currentId - The ID of the currently highlighted pass/crossing.
 * @param {object} item - The data object for the highlighted pass/crossing.
 * @param {object} mapHandler - The map handler module instance.
 * @param {object} stationsData - The main station data object.
 */
export function updateFormFromSelection(dom, selectedStations, currentId, item, mapHandler, stationsData) {
    if (!currentId || !item) return;
    const selectedCameraViews = [];
    const checkedCameras = document.querySelectorAll('input[name="cameras"]:checked');
    const currentStationId = selectedStations.values().next().value;
    document.querySelectorAll('.event-link').forEach(link => {
        const linkParent = link.closest('.satellite-group');
        if (!linkParent) return;
        const linkId = linkParent.querySelector('h6').dataset.passId || linkParent.querySelector('h6').dataset.crossingId;
        if (linkId === currentId) {
            const camNum = parseInt(link.dataset.camera, 10);
            const stationId = link.dataset.stationId;
            const isChecked = Array.from(checkedCameras).some(cb => parseInt(cb.value, 10) === camNum 
                && stationId === currentStationId);
            link.classList.toggle('selected', isChecked);
            if (isChecked) {
                const view = item.camera_views.find(cv => cv.camera === camNum && cv.station_id === stationId);
                if (view) selectedCameraViews.push(view);
         
           }
        } else {
   
 
            link.classList.remove('selected');
        }
    });
    mapHandler.clearBearingLines();
    if (selectedCameraViews.length === 0) return;

    let earliestStart = new Date(selectedCameraViews[0].start_utc);
    let latestEnd = new Date(selectedCameraViews[0].end_utc);
    selectedCameraViews.forEach(view => {
        const start = new Date(view.start_utc), end = new Date(view.end_utc);
        if (start < earliestStart) earliestStart = start;
        if (end > latestEnd) latestEnd = end;
    });
    dom.dateInput.value = earliestStart.toISOString().slice(0, 10);
    dom.hourSelect.value = earliestStart.getUTCHours();
    dom.minuteSelect.value = earliestStart.getUTCMinutes();
    const durationMinutes = (latestEnd.getTime() - earliestStart.getTime()) / (1000 * 60);
    dom.lengthSelect.value = Math.max(1, Math.ceil(durationMinutes));
    dom.intervalSelect.value = 1;
    dom.dateInput.dispatchEvent(new Event('change'));
    
    if (document.getElementById('satellite-toggle').checked || document.getElementById('aircraft-toggle').checked) {
        mapHandler.drawBearingLines(item, selectedCameraViews, stationsData);
    }
}

/**
 * Clears all camera selections in the form and related UI elements.
 * @param {object} mapHandler - The map handler module instance.
 */
export function clearSelections(mapHandler) {
    document.querySelectorAll('input[name="cameras"]').forEach(cb => cb.checked = false);
    document.querySelectorAll('.event-link').forEach(link => link.classList.remove('selected'));
    mapHandler.clearBearingLines();
}

/**
 * Toggles the visibility of a side panel and its corresponding map layer.
 * @param {string} panelType - The type of panel (e.g., 'lightning').
 * @param {boolean} isChecked - Whether the corresponding toggle is checked.
 * @param {object} mapHandler - The map handler module instance.
 */
export function togglePanelAndLayer(panelType, isChecked, mapHandler) {
    document.getElementById(`${panelType}-panel-container`).style.display = isChecked ? 'block' : 'none';
    if (panelType === 'lightning') {
        document.getElementById('lightning-filter-label').style.display = isChecked ? 'inline-flex' : 'none';
        mapHandler.toggleLayer('lightning', isChecked);
        
        // Clear highlighting when lightning is turned off
        if (!isChecked) {
            mapHandler.clearLightningHighlighting();
        }
    }
}

/**
 * Highlights a specific pass/crossing in its side panel and scrolls it into view.
 * @param {string} id - The ID of the pass or crossing to highlight.
 */
export function highlightPassInPanel(id) {
    document.querySelectorAll('.satellite-group').forEach(el => {
        const elId = el.querySelector('h6').dataset.passId || el.querySelector('h6').dataset.crossingId;
        el.classList.toggle('selected-pass', elId === id);
        if (elId === id) {
            el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    });
}

/**
 * Highlights a specific lightning strike in its side panel and scrolls it into view.
 * @param {string} strikeId - The ID of the strike list item to highlight.
 */
export function selectLightningStrikeInPanel(strikeId) {
    document.querySelectorAll('.lightning-list li.selected-lightning').forEach(el => el.classList.remove('selected-lightning'));
    const listItem = document.getElementById(strikeId);
    if (listItem) {
        listItem.classList.add('selected-lightning');
        listItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

/** Shows a side panel. */
export function showPanel(panelType) {
    document.getElementById(`${panelType}-panel-container`).style.display = 'block';
}

/** Hides a side panel.
 */
export function hidePanel(panelType) {
    document.getElementById(`${panelType}-panel-container`).style.display = 'none';
}

export function displayMeteorList(meteors, { onMeteorClick }) {
    const meteorList = document.getElementById('meteor-list');
    if (!meteorList) return;

    meteorListRenderToken += 1;
    const renderToken = meteorListRenderToken;

    if (!Array.isArray(meteors) || meteors.length === 0) {
        meteorList.replaceChildren(createEl('p', { style: 'color: #6c757d; margin: 0;', textContent: t('no_meteors_found') }));
        return;
    }

    meteorList.replaceChildren();
    const ul = createEl('ul', { className: 'meteor-list' });

    [...meteors]
        .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
        .slice(0, 200)
        .forEach((meteor) => {
            const ts = meteor && meteor.timestamp ? String(meteor.timestamp) : '';
            const label = ts ? ts.replace('T', ' ').replace('Z', ' UTC') : t('unknown_time');
            const stationCount = Array.isArray(meteor.station_ids) ? meteor.station_ids.length : 0;
            const li = createEl('li', { className: 'meteor-list-item' });
            const btn = createEl('button', {
                type: 'button',
                className: 'meteor-list-btn',
                textContent: stationCount > 0 ? `${label} (${stationCount})` : label
            });
            btn.addEventListener('click', () => onMeteorClick(meteor));

            let reportLink = null;
            const match = ts.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z$/);
            if (match) {
                const ymd = `${match[1]}${match[2]}${match[3]}`;
                const hms = `${match[4]}${match[5]}${match[6]}`;
                const url = `https://norskmeteornettverk.no/meteor/${ymd}/${hms}/`;

                const maybeLink = createEl('span', { className: 'meteor-report-link-placeholder' });
                checkUrlExists(url).then(exists => {
                    if (!exists) return;
                    if (meteorListRenderToken !== renderToken) return;
                    if (!maybeLink.isConnected) return;
                    const a = createEl('a', { href: url, target: '_blank', rel: 'noopener', className: 'meteor-report-link', textContent: t('meteor_report_link') });
                    maybeLink.replaceChildren(a);
                });
                reportLink = maybeLink;
            }

            if (reportLink) {
                const row = createEl('div', { className: 'meteor-list-row' });
                row.append(btn, reportLink);
                li.appendChild(row);
            } else {
                li.appendChild(btn);
            }
            ul.appendChild(li);
        });

    meteorList.appendChild(ul);
}

/** Displays an error message inside a panel's list area.
 */
export function showPanelError(panelType, message) {
    const listEl = document.getElementById(`${panelType}-list`);
    if (listEl) {
        listEl.replaceChildren(createEl('p', { className: 'error-msg', textContent: message }));
    }
}

/** Displays a neutral (non-error) informational message inside a panel's list area,
 * e.g. "no stations selected".
 */
export function showPanelInfo(panelType, message) {
    const listEl = document.getElementById(`${panelType}-list`);
    if (listEl) {
        listEl.replaceChildren(createEl('p', { style: 'color: #6c757d; margin: 0;', textContent: message }));
    }
}

// --- Satellite panel: time range dual-handle slider ---
// The slider spans the last SATELLITE_RANGE_MAX_HOURS hours (7 days) up to
// "now". Its integer value represents hours-since-(7-days-ago), so 0 = 7
// days ago and SATELLITE_RANGE_MAX_HOURS = now.
const SATELLITE_RANGE_MAX_HOURS = 7 * 24;

function satelliteRangeHoursToDate(hoursSinceStart) {
    return new Date(Date.now() - (SATELLITE_RANGE_MAX_HOURS - hoursSinceStart) * 3600000);
}

function formatSatelliteRangeLabel(date) {
    const datePart = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' });
    const timePart = date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'UTC' });
    return `${datePart} ${timePart} UTC`;
}

/**
 * Sets up the satellite panel's dual-handle time range slider and its
 * 1/3/7-day preset buttons. Renders day tick marks once, then keeps the
 * progress bar, labels, and active preset button in sync as the user drags
 * either handle or clicks a preset.
 * @param {object} options
 * @param {function} options.onChange - Called with `{startIso, endIso}` whenever the committed range changes.
 */
export function initSatelliteRangeSlider({ onChange }) {
    const startInput = document.getElementById('satellite-range-start');
    const endInput = document.getElementById('satellite-range-end');
    const progress = document.getElementById('satellite-range-progress');
    const startLabel = document.getElementById('satellite-range-start-label');
    const endLabel = document.getElementById('satellite-range-end-label');
    const presetButtons = document.querySelectorAll('.satellite-preset-btn');
    if (!startInput || !endInput) return;

    renderSatelliteRangeTicks();

    function updateVisuals() {
        const startVal = parseInt(startInput.value, 10);
        const endVal = parseInt(endInput.value, 10);
        const pct = v => (v / SATELLITE_RANGE_MAX_HOURS) * 100;
        if (progress) {
            progress.style.left = `${pct(startVal)}%`;
            progress.style.width = `${Math.max(0, pct(endVal) - pct(startVal))}%`;
        }
        if (startLabel) startLabel.textContent = formatSatelliteRangeLabel(satelliteRangeHoursToDate(startVal));
        if (endLabel) endLabel.textContent = formatSatelliteRangeLabel(satelliteRangeHoursToDate(endVal));
        // Keep whichever handle is on the right half of the track on top, so
        // it stays draggable even when the two handles are close together.
        if (startVal > SATELLITE_RANGE_MAX_HOURS / 2) {
            startInput.style.zIndex = 3; endInput.style.zIndex = 2;
        } else {
            startInput.style.zIndex = 2; endInput.style.zIndex = 3;
        }
        presetButtons.forEach(btn => {
            const days = parseInt(btn.dataset.days, 10);
            const expectedStart = SATELLITE_RANGE_MAX_HOURS - days * 24;
            btn.classList.toggle('active', startVal === expectedStart && endVal === SATELLITE_RANGE_MAX_HOURS);
        });
    }

    function emitChange() {
        const startVal = parseInt(startInput.value, 10);
        const endVal = parseInt(endInput.value, 10);
        if (onChange) {
            onChange({
                startIso: satelliteRangeHoursToDate(startVal).toISOString(),
                endIso: satelliteRangeHoursToDate(endVal).toISOString()
            });
        }
    }

    startInput.addEventListener('input', () => {
        if (parseInt(startInput.value, 10) > parseInt(endInput.value, 10)) startInput.value = endInput.value;
        updateVisuals();
    });
    endInput.addEventListener('input', () => {
        if (parseInt(endInput.value, 10) < parseInt(startInput.value, 10)) endInput.value = startInput.value;
        updateVisuals();
    });
    // 'change' (not 'input') fires once the user releases the handle, which
    // is when we actually want to trigger a (re)fetch.
    startInput.addEventListener('change', emitChange);
    endInput.addEventListener('change', emitChange);

    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            setSatelliteRangePreset(parseInt(btn.dataset.days, 10));
            emitChange();
        });
    });

    updateVisuals();
}

/** Programmatically moves the slider handles to match a 1/3/7-day preset. */
export function setSatelliteRangePreset(days) {
    const startInput = document.getElementById('satellite-range-start');
    const endInput = document.getElementById('satellite-range-end');
    if (!startInput || !endInput) return;
    endInput.value = SATELLITE_RANGE_MAX_HOURS;
    startInput.value = Math.max(0, SATELLITE_RANGE_MAX_HOURS - days * 24);
    // Re-trigger the same listeners 'input' would use, to refresh visuals.
    startInput.dispatchEvent(new Event('input'));
    endInput.dispatchEvent(new Event('input'));
}

/** Returns the slider's currently selected range as ISO8601 UTC bounds. */
export function getSatelliteRangeIso() {
    const startInput = document.getElementById('satellite-range-start');
    const endInput = document.getElementById('satellite-range-end');
    if (!startInput || !endInput) return null;
    return {
        startIso: satelliteRangeHoursToDate(parseInt(startInput.value, 10)).toISOString(),
        endIso: satelliteRangeHoursToDate(parseInt(endInput.value, 10)).toISOString()
    };
}

// Hour ticks are drawn every SATELLITE_TICK_HOUR_STEP hours, aligned to real
// clock boundaries (e.g. 00/06/12/18), not just every N hours from "now".
// A tick landing on 00:00 is a "major" tick showing the date; the rest are
// "minor" ticks showing just the (24-hour) hour.
const SATELLITE_TICK_HOUR_STEP = 6;

/** Returns the first UTC clock time >= `from` that's a multiple of `hourStep` hours. */
function firstAlignedTick(from, hourStep) {
    const t = new Date(from);
    t.setUTCMinutes(0, 0, 0); // Floor to the start of this UTC hour.
    if (t < from) t.setUTCHours(t.getUTCHours() + 1); // Ensure we didn't floor below `from`.
    const rem = t.getUTCHours() % hourStep;
    if (rem !== 0) t.setUTCHours(t.getUTCHours() + (hourStep - rem));
    return t;
}

function renderSatelliteRangeTicks() {
    const ticksEl = document.getElementById('satellite-range-ticks');
    if (!ticksEl) return;
    ticksEl.replaceChildren();

    const rangeStart = satelliteRangeHoursToDate(0);
    const rangeEnd = satelliteRangeHoursToDate(SATELLITE_RANGE_MAX_HOURS);

    for (let t = firstAlignedTick(rangeStart, SATELLITE_TICK_HOUR_STEP); t <= rangeEnd; t.setUTCHours(t.getUTCHours() + SATELLITE_TICK_HOUR_STEP)) {
        const hoursSinceStart = (t - rangeStart) / 3600000;
        const pct = (hoursSinceStart / SATELLITE_RANGE_MAX_HOURS) * 100;
        const isMidnight = t.getUTCHours() === 0;

        ticksEl.appendChild(createEl('div', {
            className: `satellite-range-tick${isMidnight ? ' major' : ''}`,
            style: `left: ${pct}%;`
        }));

        // Clamp the first/last labels so they don't overhang outside the slider.
        const translate = pct < 3 ? '0' : (pct > 97 ? '-100%' : '-50%');
        const label = isMidnight
            ? new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' })
            : String(t.getUTCHours()).padStart(2, '0');
        ticksEl.appendChild(createEl('div', {
            className: `satellite-range-tick-label${isMidnight ? ' major' : ''}`,
            style: `left: ${pct}%; transform: translateX(${translate});`,
            textContent: label
        }));
    }
}

/**
 * Displays or updates a progress bar inside a panel, used for async tasks like fetching passes.
 * @param {string} panelType - The type of panel ('satellite' or 'aircraft').
 * @param {object} data - The progress data from the backend.
 */
export function updateTaskProgress(panelType, data) {
    const containerId = `${panelType}-progress-container`;
    let container = document.getElementById(containerId);
    if (!container) {
        const listEl = document.getElementById(`${panelType}-list`);
        if (listEl) {
            listEl.innerHTML = `<div id="${containerId}" style="width: 95%; margin: 0 auto;"><p>${t('progress_please_wait', { message: `<span id="${panelType}-progress-text">${t('progress_calculating')}</span>`})}</p><div class="progress-bar-outline"><div id="${panelType}-progress-bar-inner" class="progress-bar-inner" style="width: 0%;"></div></div></div>`;
        }
    }
    const progressBar = document.getElementById(`${panelType}-progress-bar-inner`);
    const progressText = document.getElementById(`${panelType}-progress-text`);
    if (progressBar) progressBar.style.width = `${data.step || 0}%`;
    if (progressText) progressText.textContent = translateMessage(data.message) || t('progress_calculating');
}

/**
 * Translates a message string from the backend, which may contain a key and parameters.
 * @param {string} message - The message string, e.g., "key|param1=value1,param2=value2".
 * @returns {string} The translated string.
 */
export function translateMessage(message) {
    if (!message || typeof message !== 'string') return message;
    const parts = message.split('|');
    const key = parts[0];
    const replacements = {};
    if (parts.length > 1) {
        parts[1].split(',').forEach(param => {
            const [paramKey, paramValue] = param.split('=');
            if (paramKey && paramValue) {
                replacements[paramKey] = paramValue;
            }
        });
    }
    return t(key, replacements);
}

let stationStatsRenderToken = 0;

/**
 * Renders station observation statistics in the dedicated panel.
 * @param {object} data - The stats data returned by get_station_stats.
 * @param {object} callbacks - Object containing { onDateRangeChange, onEventClick }.
 * @param {string} startDate - Currently active start date (YYYY-MM-DD).
 * @param {string} endDate - Currently active end date (YYYY-MM-DD).
 * @param {object} [leafletMap] - Optional Leaflet map instance for drawing paths on hover.
 */
export function displayStationStats(data, { onDateRangeChange, onEventClick, onEventHover, onEventLeave }, startDate, endDate, leafletMap) {
    const container = document.getElementById('station-stats-panel-container');
    const panel = document.getElementById('station-stats-panel');
    if (!container || !panel) return;

    stationStatsRenderToken += 1;
    const renderToken = stationStatsRenderToken;

    container.style.display = 'block';

    const titleEl = panel.querySelector('h2');
    if (titleEl) titleEl.textContent = t('stats_panel_title', { station_code: data.station_code || '' });

    const listEl = document.getElementById('station-stats-list');
    if (!listEl) return;
    listEl.replaceChildren();

    const periodBar = createEl('div', { className: 'stats-period-bar' });
    const todayStr = new Date().toISOString().slice(0, 10);
    [7, 30, 90].forEach(d => {
        const presetEnd = todayStr;
        const presetStart = new Date(Date.now() - (d - 1) * 86400000).toISOString().slice(0, 10);
        const isActive = (startDate === presetStart && endDate === presetEnd);
        const btn = createEl('button', {
            type: 'button',
            className: `stats-period-btn${isActive ? ' active' : ''}`,
            textContent: t('stats_period_days', { days: d })
        });
        btn.addEventListener('click', () => onDateRangeChange(presetStart, presetEnd));
        periodBar.appendChild(btn);
    });
    const stepDate = (hiddenInput, displayInput, delta) => {
        if (!hiddenInput.value) return;
        const [y, m, d] = hiddenInput.value.split('-').map(Number);
        if (!y || !m || !d) return;
        const next = new Date(Date.UTC(y, m - 1, d + delta));
        const iso = next.toISOString().slice(0, 10);
        hiddenInput.value = iso;
        displayInput.value = iso;
    };
    const fromLabel = createEl('span', { className: 'stats-date-label', textContent: t('stats_from') });
    const fromHidden = createEl('input', { type: 'date', style: 'position:absolute;opacity:0;pointer-events:none;width:0;height:0;', value: startDate || '' });
    const fromDisplay = createEl('input', { type: 'text', className: 'stats-date-input', value: startDate || '', placeholder: 'YYYY-MM-DD', readOnly: true });
    fromDisplay.addEventListener('click', () => { try { fromHidden.showPicker(); } catch (e) { fromHidden.click(); } });
    fromHidden.addEventListener('change', () => { fromDisplay.value = fromHidden.value; });
    const fromPrev = createEl('button', { type: 'button', className: 'date-nav-btn', textContent: '\u2039' });
    const fromNext = createEl('button', { type: 'button', className: 'date-nav-btn', textContent: '\u203A' });
    fromPrev.addEventListener('click', () => stepDate(fromHidden, fromDisplay, -1));
    fromNext.addEventListener('click', () => stepDate(fromHidden, fromDisplay, 1));
    const toLabel = createEl('span', { className: 'stats-date-label', textContent: t('stats_to') });
    const toHidden = createEl('input', { type: 'date', style: 'position:absolute;opacity:0;pointer-events:none;width:0;height:0;', value: endDate || '' });
    const toDisplay = createEl('input', { type: 'text', className: 'stats-date-input', value: endDate || '', placeholder: 'YYYY-MM-DD', readOnly: true });
    toDisplay.addEventListener('click', () => { try { toHidden.showPicker(); } catch (e) { toHidden.click(); } });
    toHidden.addEventListener('change', () => { toDisplay.value = toHidden.value; });
    const toPrev = createEl('button', { type: 'button', className: 'date-nav-btn', textContent: '\u2039' });
    const toNext = createEl('button', { type: 'button', className: 'date-nav-btn', textContent: '\u203A' });
    toPrev.addEventListener('click', () => stepDate(toHidden, toDisplay, -1));
    toNext.addEventListener('click', () => stepDate(toHidden, toDisplay, 1));
    const goBtn = createEl('button', { type: 'button', className: 'stats-date-go-btn', textContent: t('stats_go') });
    goBtn.addEventListener('click', () => {
        if (fromHidden.value && toHidden.value) onDateRangeChange(fromHidden.value, toHidden.value);
    });
    periodBar.append(fromLabel, fromPrev, fromHidden, fromDisplay, fromNext, toLabel, toPrev, toHidden, toDisplay, toNext, goBtn);
    listEl.appendChild(periodBar);

    if (data.error) {
        const errMsg = t(data.error) !== data.error ? t(data.error) : data.error;
        listEl.appendChild(createEl('p', { className: 'error-msg', textContent: errMsg }));
        return;
    }

    const summary = createEl('div', { className: 'stats-summary' });
    summary.appendChild(createEl('span', { className: 'stats-total', textContent: t('stats_total_observations', { count: data.total }) }));
    summary.appendChild(createEl('span', { className: 'stats-multi', textContent: t('stats_multi_station', { count: data.multi }) }));
    summary.appendChild(createEl('span', { className: 'stats-single', textContent: t('stats_single_station', { count: data.total - data.multi }) }));
    if (data.has_trajectory_details) {
        const orbitTotal = data.shower_count + data.sporadic_count;
        if (orbitTotal > 0) {
            summary.appendChild(createEl('span', { className: 'stats-orbit', textContent: t('stats_orbit_count', { count: orbitTotal }) }));
            summary.appendChild(createEl('span', { className: 'stats-shower', textContent: t('stats_shower_count', { count: data.shower_count }) }));
            summary.appendChild(createEl('span', { className: 'stats-sporadic', textContent: t('stats_sporadic_count', { count: data.sporadic_count }) }));
        }
        if (data.avg_speed != null) {
            summary.appendChild(createEl('span', { className: 'stats-speed', textContent: t('stats_avg_speed', { speed: data.avg_speed }) }));
        }
        if (data.median_speed != null) {
            summary.appendChild(createEl('span', { className: 'stats-speed', textContent: t('stats_median_speed', { speed: data.median_speed }) }));
        }
        if (data.avg_start_alt != null) {
            summary.appendChild(createEl('span', { className: 'stats-alt', textContent: t('stats_start_alt_summary', { avg: data.avg_start_alt, median: data.median_start_alt }) }));
        }
        if (data.avg_end_alt != null) {
            summary.appendChild(createEl('span', { className: 'stats-alt', textContent: t('stats_end_alt_summary', { avg: data.avg_end_alt, median: data.median_end_alt }) }));
        }
    }
    listEl.appendChild(summary);

    if (!data.events || data.events.length === 0) {
        listEl.appendChild(createEl('p', { style: 'color: #6c757d; margin: 8px 0 0;', textContent: t('stats_no_events') }));
        return;
    }

    const ul = createEl('ul', { className: 'stats-event-list' });
    data.events.forEach(event => {
        const li = createEl('li', { className: `stats-event-item${event.num_stations > 1 ? ' multi' : ''}` });

        const ts = event.timestamp || '';
        const label = ts ? ts.replace('T', ' ').replace('Z', ' UTC') : t('unknown_time');

        const btn = createEl('button', {
            type: 'button',
            className: 'stats-event-btn',
        });
        const timeSpan = createEl('span', { className: 'stats-event-time', textContent: label });
        const countBadge = createEl('span', {
            className: `stats-station-badge${event.num_stations > 1 ? ' multi' : ''}`,
            textContent: event.num_stations > 1 ? `${event.num_stations} ${t('stats_stations_short')}` : `1 ${t('stats_station_short')}`
        });
        btn.append(timeSpan, countBadge);
        btn.addEventListener('click', () => {
            if (onEventClick) onEventClick(event.timestamp);
        });

        const row = createEl('div', { className: 'stats-event-row' });
        row.appendChild(btn);

        const match = ts.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z$/);
        if (match) {
            const ymd = `${match[1]}${match[2]}${match[3]}`;
            const hms = `${match[4]}${match[5]}${match[6]}`;
            const url = `https://norskmeteornettverk.no/meteor/${ymd}/${hms}/`;
            row.appendChild(createEl('a', { href: url, target: '_blank', rel: 'noopener', className: 'meteor-report-link', textContent: t('meteor_report_link') }));
        }

        li.appendChild(row);

        const detailParts = [];
        if (event.other_stations && event.other_stations.length > 0) {
            detailParts.push(`${t('stats_also_seen_at')}: ${event.other_stations.join(', ')}`);
        }
        if (event.shower) {
            const showerLabel = event.shower.toLowerCase() === 'sporadic' ? t('stats_sporadic') : event.shower;
            detailParts.push(`${t('stats_shower_label')}: ${showerLabel}`);
        }
        if (event.speed != null) {
            detailParts.push(`${t('stats_speed_label')}: ${event.speed} km/s`);
        }
        if (event.direction != null) {
            detailParts.push(`${t('stats_direction_label')}: ${event.direction}°`);
        }
        if (event.start_alt != null) {
            detailParts.push(`${t('stats_alt_label')}: ${event.start_alt} → ${event.end_alt} km`);
        }
        if (detailParts.length > 0) {
            li.appendChild(createEl('div', { className: 'stats-event-detail', textContent: detailParts.join('  ·  ') }));
        }
        const allStationCodes = [data.station_code, ...(event.other_stations || [])];
        if (event.num_stations > 1 && onEventHover) {
            li.addEventListener('mouseenter', () => onEventHover(allStationCodes));
            li.addEventListener('mouseleave', () => onEventLeave(allStationCodes));
        }
        if (event.start_lat != null && event.end_lat != null && leafletMap) {
            li.addEventListener('mouseenter', () => {
                if (window._statsPathLayer) { window._statsPathLayer.remove(); window._statsPathLayer = null; }
                const zoom = leafletMap.getZoom();
                const metersPerPixel = 40075016.686 * Math.abs(Math.cos(leafletMap.getCenter().lat * Math.PI / 180)) / Math.pow(2, zoom + 8);
                const h1 = event.start_alt != null ? event.start_alt : 80;
                const h2 = event.end_alt != null ? event.end_alt : 40;
                const getW = (h, isEnd) => (1 + (1 - (Math.min(100, Math.max(0, h)) / 100)) * 4 + (isEnd ? 3 : 0)) * metersPerPixel;
                const w1 = getW(h1, false), w2 = getW(h2, true);
                const bearing = calculateBearing(event.start_lat, event.start_lon, event.end_lat, event.end_lon);
                const perp1 = (bearing + 90) % 360, perp2 = (bearing - 90 + 360) % 360;
                const p1L = destinationPoint(event.start_lat, event.start_lon, w1 / 2, perp2);
                const p1R = destinationPoint(event.start_lat, event.start_lon, w1 / 2, perp1);
                const p2L = destinationPoint(event.end_lat, event.end_lon, w2 / 2, perp2);
                const p2R = destinationPoint(event.end_lat, event.end_lon, w2 / 2, perp1);
                const poly = L.polygon([p1L, p1R, p2R, p2L], { color: '#ff9900', fillColor: '#ff9900', weight: 0, fillOpacity: 0.7 });
                const endCap = L.circle([event.end_lat, event.end_lon], { radius: w2 / 2, color: '#ff9900', fillColor: '#ff9900', weight: 0, fillOpacity: 0.7 });
                window._statsPathLayer = L.featureGroup([poly, endCap]).addTo(leafletMap);
            });
            li.addEventListener('mouseleave', () => {
                if (window._statsPathLayer) { window._statsPathLayer.remove(); window._statsPathLayer = null; }
            });
        }
        ul.appendChild(li);
    });
    listEl.appendChild(ul);
}

/**
 * Hides the station statistics panel.
 */
export function hideStationStats() {
    const container = document.getElementById('station-stats-panel-container');
    if (container) container.style.display = 'none';
}

/**
 * Shows a loading state in the station stats panel.
 * @param {string} stationCode - The station code to display in the title.
 */
export function showStationStatsLoading(stationCode) {
    const container = document.getElementById('station-stats-panel-container');
    const panel = document.getElementById('station-stats-panel');
    if (!container || !panel) return;
    container.style.display = 'block';
    const titleEl = panel.querySelector('h2');
    if (titleEl) titleEl.textContent = t('stats_panel_title', { station_code: stationCode });
    const listEl = document.getElementById('station-stats-list');
    if (listEl) listEl.replaceChildren(createEl('p', { style: 'color: #6c757d; margin: 0;', textContent: t('stats_loading') }));
}
