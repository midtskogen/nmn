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
 * Parses a filename and builds an enhanced title with station info, coordinates, elevation, sun altitude, and ISO timestamp.
 * Filename formats:
 * - Regular: stationCode_camN_YYYYMMDD_HHMM_type.ext
 * - Stitched: stationCode_YYYYMMDD_HHMM_resolution_projection.ext (no camN)
 * @param {string} filename - The filename to parse
 * @returns {string} The enhanced title
 */
function buildEnhancedPreviewTitle(filename) {
    if (!previewStationsData) return filename;

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
        sunAltText = ` | ${t('sun_altitude', 'Sun')}: ${sunAlt.toFixed(1)}°`;
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
    document.querySelectorAll('input[name="cameras"]').forEach(cb => cb.checked = true);
    document.querySelector('input[name="primary_file_type"][value="image"]').checked = true;
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
                high: t('altitude_quality_high', 'High accuracy: GPS altitude'),
                medium: t('altitude_quality_medium', 'Medium accuracy: Mix of GPS and barometric'),
                low: t('altitude_quality_low', 'Low accuracy: Primarily barometric altitude')
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
    const enhancedTitle = buildEnhancedPreviewTitle(title);

    // Header with title and close button
    const header = createEl('div', { className: 'preview-header' });
    header.appendChild(createEl('h3', { textContent: enhancedTitle, className: 'preview-title' }));
    const closeButton = createEl('button', { className: 'preview-close-btn', textContent: '×' });
    header.appendChild(closeButton);

    // Video container with overlay for timestamp
    const videoWrapper = createEl('div', { className: 'preview-video-wrapper' });
    const video = createEl('video', {
        src: videoUrl,
        className: 'preview-video',
        controls: false,
        preload: 'metadata',
        autoplay: true,
        muted: true
    });

    // Grid and annotation overlays for archive videos (hidden by default via opacity, shown via toggle)
    const gridOverlay = createEl('img', { id: 'grid-overlay-image', className: 'archive-overlay grid-overlay', style: { display: 'block', position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 10 } });
    const annotationOverlay = createEl('img', { id: 'annotation-overlay-image', className: 'archive-overlay annotation-overlay', style: { display: 'block', position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 11 } });
    // Explicitly set opacity 0 to hide initially
    gridOverlay.style.opacity = '0';
    annotationOverlay.style.opacity = '0';

    // Timestamp overlay with date (2 decimal precision) - lower right
    const timestampOverlay = createEl('div', { className: 'preview-timestamp', textContent: '' });

    // Loading indicator
    const loadingIndicator = createEl('div', { className: 'preview-loading', textContent: t('loading', 'Loading...') });

    videoWrapper.append(video, gridOverlay, annotationOverlay, timestampOverlay, loadingIndicator);

    // Playback controls
    const controls = createEl('div', { className: 'preview-controls' });

    // Play/Pause button - video autoplays so start with pause symbol
    const playPauseBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '⏸',
        title: t('modal_play_pause', 'Play/Pause')
    });

    // Frame step controls
    const frameBackBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '◀',
        title: t('modal_frame_back', 'Previous Frame')
    });

    const frameForwardBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '▶',
        title: t('modal_frame_forward', 'Next Frame')
    });

    // Rewind button
    const rewindBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '⏮',
        title: t('modal_rewind', 'Rewind to Start')
    });

    // Screenshot button
    const screenshotBtn = createEl('button', {
        className: 'preview-control-btn screenshot',
        textContent: '📷',
        title: t('screenshot', 'Take Screenshot')
    });

    // Download button
    const downloadBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '⬇',
        title: t('download_video', 'Download Video')
    });

    // Fullscreen button
    const fullscreenBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '⛶',
        title: t('fullscreen', 'Fullscreen')
    });

    // Navigation buttons (prev/next) - shown when mediaList is provided
    // Always create buttons even with 1 item, since more may load later
    // Buttons not disabled - click handlers check bounds dynamically using currentMediaList
    let prevBtn = null, nextBtn = null, navInfo = null;
    if (mediaList && mediaList.length > 0) {
        prevBtn = createEl('button', {
            className: 'preview-control-btn nav-btn',
            textContent: '◀',
            title: t('previous', 'Previous')
        });
        nextBtn = createEl('button', {
            className: 'preview-control-btn nav-btn',
            textContent: '▶',
            title: t('next', 'Next')
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
        title: t('brightness', 'Brightness'),
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
        title: t('contrast', 'Contrast'),
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
        title: t('saturation', 'Saturation'),
        id: 'saturation-slider'
    });

    // Reset filters button
    const resetFiltersBtn = createEl('button', {
        className: 'preview-control-btn reset',
        textContent: t('reset_filters', 'Reset'),
        title: t('reset_filters', 'Reset to default')
    });

    // Timestamp toggle checkbox
    const timestampToggleContainer = createEl('label', { className: 'preview-timestamp-toggle' });
    const timestampCheckbox = createEl('input', {
        type: 'checkbox',
        checked: true
    });
    timestampToggleContainer.append(timestampCheckbox, ' ', t('show_timestamp', 'Show timestamp'));

    // Parse station and camera from video filename (e.g., "GAU_cam1_20260429_2056_hires.mp4")
    const filenameMatch = title.match(/^([A-Z]{3})_cam(\d+)_\d{8}_\d{4}/);
    let stationId = null, cameraNum = null, videoTimestamp = null, annotationTimestamp = null;
    let gridToggleContainer = null, annotationToggleContainer = null;
    let gridCheckbox = null, annotationCheckbox = null;

    if (filenameMatch) {
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
        gridToggleContainer.append(gridCheckbox, ' ', t('modal_grid_toggle', 'Show Grid'));

        // Annotation overlay toggle - initially greyed out until loaded
        annotationToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        annotationCheckbox = createEl('input', { type: 'checkbox', id: 'annotation-overlay-toggle', disabled: true });
        annotationToggleContainer.append(annotationCheckbox, ' ', t('modal_annotation_toggle', 'Show Stars'));
    }

    // Brightness: label above slider in a small inline column
    const brightnessWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    brightnessWrapper.append(
        createEl('label', { textContent: t('brightness', 'Brightness'), htmlFor: 'brightness-slider', className: 'preview-filter-label' }),
        brightnessSlider
    );

    // Contrast: label above slider in a small inline column
    const contrastWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    contrastWrapper.append(
        createEl('label', { textContent: t('contrast', 'Contrast'), htmlFor: 'contrast-slider', className: 'preview-filter-label' }),
        contrastSlider
    );

    // Saturation: label above slider in a small inline column
    const saturationWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    saturationWrapper.append(
        createEl('label', { textContent: t('saturation', 'Saturation'), htmlFor: 'saturation-slider', className: 'preview-filter-label' }),
        saturationSlider
    );

    const checkboxesWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '4px' } });
    checkboxesWrapper.append(timestampToggleContainer);
    if (gridToggleContainer) checkboxesWrapper.append(gridToggleContainer);
    if (annotationToggleContainer) checkboxesWrapper.append(annotationToggleContainer);

    filterControls.append(resetFiltersBtn, brightnessWrapper, contrastWrapper, saturationWrapper, checkboxesWrapper);

    // Assemble modal
    modalContent.append(header, videoWrapper, controls, filterControls);
    modalBackdrop.appendChild(modalContent);
    document.body.appendChild(modalBackdrop);

    // Push history state for back button handling
    history.pushState({ modalOpen: true }, '');

    // Load grid overlay - fetch JSON metadata first, then set image src
    if (videoTimestamp && stationId && cameraNum) {
        const gridApiUrl = `index.php?action=fetch_archive_grid&station_id=${stationId}&camera_num=${cameraNum}&timestamp=${encodeURIComponent(videoTimestamp)}`;
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
            .catch(err => {
                gridToggleContainer.style.opacity = '0.5';
                gridCheckbox.disabled = true;
            });

        // Grid toggle handler - toggle opacity (0.6)
        gridCheckbox.addEventListener('change', () => {
            gridOverlay.style.opacity = gridCheckbox.checked ? '0.6' : '0';
        });

        // Load annotation overlay - fetch JSON metadata first, then set image src
        const annotationApiUrl = `index.php?action=fetch_archive_annotation&station_id=${stationId}&camera_num=${cameraNum}&timestamp=${encodeURIComponent(annotationTimestamp)}`;
        fetch(annotationApiUrl)
            .then(response => response.json())
            .then(data => {
                if (data.success && data.annotation_url) {
                    annotationOverlay.src = data.annotation_url;
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

    // Video event handlers
    let isPlaying = true; // Video autoplays so start as playing
    let frameStep = 1 / 30; // Assume 30fps, will be updated when metadata loads

    // Get current date once at initialization
    const currentDateStr = new Date().toISOString().substr(0, 10);

    // Helper to format timestamp with actual video date
    // When seconds is 0 (not started), use current date instead of epoch (1970)
    function getFormattedTimestamp(seconds) {
        const effectiveSeconds = seconds || 0;
        // When video hasn't started, show current date with 00:00:00.00
        if (effectiveSeconds === 0) {
            return `${currentDateStr} 00:00:00.00`;
        }
        // For playing video, use epoch time but only take the time portion
        const date = new Date(effectiveSeconds * 1000);
        const timeStr = date.toISOString().substr(11, 8); // HH:MM:SS
        const decimals = String(Math.floor((effectiveSeconds % 1) * 100)).padStart(2, '0');
        return `${currentDateStr} ${timeStr}.${decimals}`;
    }

    // Set initial timestamp immediately before video loads
    timestampOverlay.textContent = `${currentDateStr} 00:00:00.00`;

    video.addEventListener('loadedmetadata', () => {
        loadingIndicator.style.display = 'none';
        // Clear fixed dimensions to allow natural video sizing
        if (initialDimensions) {
            modalContent.style.width = '';
            modalContent.style.height = '';
            modalContent.style.minWidth = '';
            modalContent.style.minHeight = '';
        }
        // Try to detect frame rate from video or default to 30
        frameStep = 1 / 30;
        // Advance one frame to get valid currentTime (avoid 1970 epoch issue)
        video.currentTime = frameStep;
        // Trigger timeupdate to refresh timestamp
        const event = new Event('timeupdate');
        video.dispatchEvent(event);
    });

    video.addEventListener('timeupdate', () => {
        // Update timestamp overlay with date and 2-decimal precision (lower right)
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

    // Control handlers
    playPauseBtn.addEventListener('click', () => {
        if (isPlaying) {
            video.pause();
            playPauseBtn.textContent = '▶';
        } else {
            video.play();
            playPauseBtn.textContent = '⏸';
        }
        isPlaying = !isPlaying;
    });

    frameBackBtn.addEventListener('click', () => {
        video.pause();
        isPlaying = false;
        playPauseBtn.textContent = '▶';
        video.currentTime = Math.max(0, video.currentTime - frameStep);
    });

    frameForwardBtn.addEventListener('click', () => {
        video.pause();
        isPlaying = false;
        playPauseBtn.textContent = '▶';
        video.currentTime = Math.min(video.duration, video.currentTime + frameStep);
    });

    // Rewind button handler
    rewindBtn.addEventListener('click', () => {
        video.currentTime = 0;
        if (!isPlaying) {
            video.play();
            isPlaying = true;
            playPauseBtn.textContent = '⏸';
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

    // Reset filters button handler
    resetFiltersBtn.addEventListener('click', () => {
        brightnessSlider.value = 1;
        contrastSlider.value = 1;
        saturationSlider.value = 1;
        updateFilters();
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

    // --- Pan/Zoom Logic (same as live video) ---
    let scale=1, panX=0, panY=0, isPanning=false, startPanX=0, startPanY=0, panOriginX=0, panOriginY=0;
    const clamp = (val, min, max) => Math.min(Math.max(val, min), max);
    const updateTransform = () => { 
        const transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
        video.style.transform = gridOverlay.style.transform = annotationOverlay.style.transform = transform;
    };
    video.style.transformOrigin = gridOverlay.style.transformOrigin = annotationOverlay.style.transformOrigin = '0 0';
    const onWheel = e => { e.preventDefault(); const rect=videoWrapper.getBoundingClientRect(); const videoRect=video.getBoundingClientRect(); const mouseX=e.clientX-rect.left; const mouseY=e.clientY-rect.top; const newScale=clamp(scale*(e.deltaY>0?0.9:1.1),1,8); const newPanX=mouseX-(mouseX-panX)*(newScale/scale); const newPanY=mouseY-(mouseY-panY)*(newScale/scale); scale=newScale; if(scale<=1.01){panX=0;panY=0;}else{panX=clamp(newPanX,-(videoRect.width*(scale-1)),0); panY=clamp(newPanY,-(videoRect.height*(scale-1)),0);} updateTransform(); };
    const onMouseMove = e => { if(!isPanning)return; const videoRect=video.getBoundingClientRect(); panX=clamp(startPanX+(e.clientX-panOriginX),-(videoRect.width*(scale-1)),0); panY=clamp(startPanY+(e.clientY-panOriginY),-(videoRect.height*(scale-1)),0); updateTransform(); };
    const onMouseUp = () => { isPanning=false; videoWrapper.style.cursor='default'; window.removeEventListener('mousemove',onMouseMove); window.removeEventListener('mouseup',onMouseUp); };
    const onMouseDown = e => { if(e.button!==0)return; e.preventDefault(); isPanning=true; videoWrapper.style.cursor='grabbing'; panOriginX=e.clientX; panOriginY=e.clientY; startPanX=panX; startPanY=panY; window.addEventListener('mousemove',onMouseMove); window.addEventListener('mouseup',onMouseUp); };
    videoWrapper.addEventListener('wheel', onWheel); videoWrapper.addEventListener('mousedown', onMouseDown);
    
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
        }
        gridOverlay.style.opacity = gridCheckbox?.checked ? '0.6' : '0';
        annotationOverlay.style.opacity = annotationCheckbox?.checked ? '0.6' : '0';
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
        document.removeEventListener('fullscreenchange', onFullscreenChange);
        videoWrapper.removeEventListener('wheel', onWheel);
        videoWrapper.removeEventListener('mousedown', onMouseDown);
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
    const enhancedTitle = buildEnhancedPreviewTitle(title);

    // Header
    const header = createEl('div', { className: 'preview-header' });
    header.appendChild(createEl('h3', { textContent: enhancedTitle, className: 'preview-title' }));
    const closeButton = createEl('button', { className: 'preview-close-btn', textContent: '×' });
    header.appendChild(closeButton);

    // Image wrapper with pan/zoom
    const imageWrapper = createEl('div', { className: 'preview-video-wrapper' });
    const img = createEl('img', {
        src: imageUrl,
        className: 'preview-video',
        style: { display: 'block', width: '100%', height: 'auto', objectFit: 'contain' }
    });

    // Grid and annotation overlays (hidden by default)
    const gridOverlay = createEl('img', { className: 'archive-overlay grid-overlay', style: { display: 'block', position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 10 } });
    const annotationOverlay = createEl('img', { className: 'archive-overlay annotation-overlay', style: { display: 'block', position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 11 } });
    gridOverlay.style.opacity = '0';
    annotationOverlay.style.opacity = '0';

    const loadingIndicator = createEl('div', { className: 'preview-loading', textContent: t('loading', 'Loading...') });
    imageWrapper.append(img, gridOverlay, annotationOverlay, loadingIndicator);

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
        title: t('download_image', 'Download Image')
    });
    const fullscreenBtn = createEl('button', {
        className: 'preview-control-btn',
        textContent: '⛶',
        title: t('fullscreen', 'Fullscreen')
    });

    // Navigation buttons (prev/next) - shown when mediaList is provided
    // Always create buttons even with 1 item, since more may load later
    // Buttons not disabled - click handlers check bounds dynamically using currentMediaList
    let prevBtn = null, nextBtn = null, navInfo = null;
    if (mediaList && mediaList.length > 0) {
        prevBtn = createEl('button', {
            className: 'preview-control-btn nav-btn',
            textContent: '◀',
            title: t('previous', 'Previous')
        });
        nextBtn = createEl('button', {
            className: 'preview-control-btn nav-btn',
            textContent: '▶',
            title: t('next', 'Next')
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
    const resetFiltersBtn = createEl('button', { className: 'preview-control-btn reset', textContent: t('reset_filters', 'Reset'), title: t('reset_filters', 'Reset to default') });

    const brightnessWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    brightnessWrapper.append(createEl('label', { textContent: t('brightness', 'Brightness'), htmlFor: 'img-brightness-slider', className: 'preview-filter-label' }), brightnessSlider);
    const contrastWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    contrastWrapper.append(createEl('label', { textContent: t('contrast', 'Contrast'), htmlFor: 'img-contrast-slider', className: 'preview-filter-label' }), contrastSlider);
    const saturationWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    saturationWrapper.append(createEl('label', { textContent: t('saturation', 'Saturation'), htmlFor: 'img-saturation-slider', className: 'preview-filter-label' }), saturationSlider);

    // Parse station and camera from image filename (e.g., "GAU_cam1_20260429_2056_image.jpg")
    const filenameMatch = title.match(/^([A-Z]{3})_cam(\d+)_(\d{8})_(\d{4})/);
    let gridToggleContainer = null, annotationToggleContainer = null;
    let gridCheckbox = null, annotationCheckbox = null;

    if (filenameMatch) {
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
        gridToggleContainer.append(gridCheckbox, ' ', t('modal_grid_toggle', 'Show Grid'));

        // Annotation overlay toggle - initially greyed out until loaded
        annotationToggleContainer = createEl('label', { className: 'preview-overlay-toggle', style: { opacity: '0.5' } });
        annotationCheckbox = createEl('input', { type: 'checkbox', id: 'img-annotation-overlay-toggle', disabled: true });
        annotationToggleContainer.append(annotationCheckbox, ' ', t('modal_annotation_toggle', 'Show Stars'));

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
    if (annotationToggleContainer) checkboxesWrapper.append(annotationToggleContainer);

    filterControls.append(resetFiltersBtn, brightnessWrapper, contrastWrapper, saturationWrapper);
    if (checkboxesWrapper.hasChildNodes()) filterControls.append(checkboxesWrapper);

    function updateFilters() {
        img.style.filter = `brightness(${brightnessSlider.value}) contrast(${contrastSlider.value}) saturate(${saturationSlider.value})`;
    }
    brightnessSlider.addEventListener('input', updateFilters);
    contrastSlider.addEventListener('input', updateFilters);
    saturationSlider.addEventListener('input', updateFilters);
    resetFiltersBtn.addEventListener('click', () => { brightnessSlider.value = 1; contrastSlider.value = 1; saturationSlider.value = 1; updateFilters(); });

    // Assemble modal
    modalContent.append(header, imageWrapper, controls, filterControls);
    modalBackdrop.appendChild(modalContent);
    document.body.appendChild(modalBackdrop);

    // Push history state for back button handling
    history.pushState({ modalOpen: true }, '');

    // Pan/Zoom
    let scale = 1, panX = 0, panY = 0, isPanning = false, startPanX = 0, startPanY = 0, panOriginX = 0, panOriginY = 0;
    const clamp = (val, min, max) => Math.min(Math.max(val, min), max);
    const updateTransform = () => {
        const transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
        img.style.transform = gridOverlay.style.transform = annotationOverlay.style.transform = transform;
    };
    img.style.transformOrigin = gridOverlay.style.transformOrigin = annotationOverlay.style.transformOrigin = 'center center';

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
        const newScale = clamp(scale * (e.deltaY > 0 ? 0.9 : 1.1), 1, 8);
        const newPanX = mouseX - (mouseX - panX) * (newScale / scale);
        const newPanY = mouseY - (mouseY - panY) * (newScale / scale);
        scale = newScale;
        if (scale <= 1.01) { panX = 0; panY = 0; } else {
            const maxPanX = baseImgWidth * (scale - 1) / 2;
            const maxPanY = baseImgHeight * (scale - 1) / 2;
            panX = clamp(newPanX, -maxPanX, maxPanX);
            panY = clamp(newPanY, -maxPanY, maxPanY);
        }
        updateTransform();
    };
    const onMouseMove = e => { if (!isPanning) return; const rect = imageWrapper.getBoundingClientRect(); const imgNaturalWidth = img.naturalWidth || img.offsetWidth; const imgNaturalHeight = img.naturalHeight || img.offsetHeight; const wrapperAspect = rect.width / rect.height; const imgAspect = imgNaturalWidth / imgNaturalHeight; let baseImgWidth, baseImgHeight; if (imgAspect > wrapperAspect) { baseImgWidth = rect.width; baseImgHeight = rect.width / imgAspect; } else { baseImgWidth = rect.height * imgAspect; baseImgHeight = rect.height; } const maxPanX = baseImgWidth * (scale - 1) / 2; const maxPanY = baseImgHeight * (scale - 1) / 2; panX = clamp(startPanX + (e.clientX - panOriginX), -maxPanX, maxPanX); panY = clamp(startPanY + (e.clientY - panOriginY), -maxPanY, maxPanY); updateTransform(); };
    const onMouseUp = () => { isPanning = false; imageWrapper.style.cursor = 'default'; window.removeEventListener('mousemove', onMouseMove); window.removeEventListener('mouseup', onMouseUp); };
    const onMouseDown = e => { if (e.button !== 0) return; e.preventDefault(); isPanning = true; imageWrapper.style.cursor = 'grabbing'; panOriginX = e.clientX; panOriginY = e.clientY; startPanX = panX; startPanY = panY; window.addEventListener('mousemove', onMouseMove); window.addEventListener('mouseup', onMouseUp); };
    imageWrapper.addEventListener('wheel', onWheel); imageWrapper.addEventListener('mousedown', onMouseDown);

    // Fullscreen
    fullscreenBtn.addEventListener('click', () => {
        if (imageWrapper.requestFullscreen) imageWrapper.requestFullscreen();
        else if (imageWrapper.webkitRequestFullscreen) imageWrapper.webkitRequestFullscreen();
    });
    const onFullscreenChange = () => {
        if (!document.fullscreenElement) { scale = 1; panX = 0; panY = 0; updateTransform(); }
        gridOverlay.style.opacity = gridCheckbox?.checked ? '0.6' : '0';
        annotationOverlay.style.opacity = annotationCheckbox?.checked ? '0.6' : '0';
    };
    document.addEventListener('fullscreenchange', onFullscreenChange);

    // Download
    downloadBtn.addEventListener('click', () => {
        const link = document.createElement('a');
        link.href = imageUrl;
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
    closeButton.addEventListener('click', () => {
        document.removeEventListener('fullscreenchange', onFullscreenChange);
        imageWrapper.removeEventListener('wheel', onWheel);
        imageWrapper.removeEventListener('mousedown', onMouseDown);
        history.back();
    });

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
                        li.appendChild(createEl('a', { href: file.url, target: '_blank', textContent: file.name, title: file.name }));
                    }

               
                     const linksContainer = createEl('div', { className: 'alternate-links' });
                    const allFilesForThisThumb = [{ url: file.url, name: file.name }, ...(file.alternatives || [])];
                    const preferredLinks = {};
                    
                    const getShortName = (filename) => {
                        if (filename.includes('_image_long_stacked.jpg')) return 'bhL';
                        if (filename.includes('_image_lowres_long_stacked.jpg')) return 'blL';
                        if (filename.includes('_hires_fisheye.jpg')) return 'fe';
                        if (filename.includes('_lowres_fisheye.jpg')) return 'fe';
                        if (filename.includes('_hires_equirect.jpg')) return 'eq';
                        if (filename.includes('_lowres_equirect.jpg')) return 'eq';
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

                    allFilesForThisThumb.forEach(f => {
                        const shortName = getShortName(f.name);
                        const isHevc = f.name.includes('_hevc.mp4');
                        const existing = preferredLinks[shortName];
                
 
                        if (!existing) {
                            preferredLinks[shortName] = f;
                       
                    
                         
                        } else {
                            const existingIsHevc = existing.name.includes('_hevc.mp4');
                            if (hevcSupported) { if (isHevc && !existingIsHevc) preferredLinks[shortName] = f; }
    
                                                   
                            else { if (!isHevc && existingIsHevc) preferredLinks[shortName] = f; }
                        }
                 
                   });
                    Object.entries(preferredLinks).sort((a, b) => a[0].localeCompare(b[0])).forEach(([shortName, linkInfo]) => {
                        const linkEl = createEl('a', { href: '#', textContent: shortName });
                        linkEl.addEventListener('click', (e) => {
                            e.preventDefault();
                            // Find this link's index in media list
                            const linkIndex = currentMediaList.findIndex(m => m.url === linkInfo.url && m.name === linkInfo.name);
                            if (linkInfo.url.endsWith('.mp4')) {
                                showVideoPreview(linkInfo.url, linkInfo.name, currentMediaList, linkIndex >= 0 ? linkIndex : mediaIndex);
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
        sunAltText = ` | ${t('sun_altitude', 'Sun')}: ${sunAlt.toFixed(1)}°`;
    }

    // Build title with coordinates, elevation, and sun altitude
    let titleText = `${displayName} – ${cameraNum}`;
    if (astronomy) {
        const lat = `${astronomy.latitude.toFixed(3)}N`;
        const lon = `${astronomy.longitude.toFixed(3)}E`;
        const elev = astronomy.elevation ? `${astronomy.elevation}m` : '';
        titleText += ` (${lat}, ${lon}${elev ? `, ${elev}` : ''}${sunAltText})`;
    }
    const modalTitle = createEl('h3', { id: 'video-modal-title', textContent: titleText });
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
    const closeButton = createEl('button', { id: 'video-close-button', textContent: t('modal_close_button'), onclick: hideVideoModal });

    // Filter controls
    const liveFilterControls = createEl('div', { className: 'preview-filter-controls' });
    const liveBrightnessSlider = createEl('input', { type: 'range', min: '0.5', max: '2', step: '0.1', value: '1', className: 'preview-slider', id: 'live-brightness-slider' });
    const liveContrastSlider = createEl('input', { type: 'range', min: '0.5', max: '2', step: '0.1', value: '1', className: 'preview-slider', id: 'live-contrast-slider' });
    const liveSaturationSlider = createEl('input', { type: 'range', min: '0', max: '3', step: '0.1', value: '1', className: 'preview-slider', id: 'live-saturation-slider' });
    const liveResetBtn = createEl('button', { className: 'preview-control-btn reset', textContent: t('reset_filters', 'Reset'), title: t('reset_filters', 'Reset to default') });
    const liveBrightnessWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    liveBrightnessWrapper.append(createEl('label', { textContent: t('brightness', 'Brightness'), htmlFor: 'live-brightness-slider', className: 'preview-filter-label' }), liveBrightnessSlider);
    const liveContrastWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    liveContrastWrapper.append(createEl('label', { textContent: t('contrast', 'Contrast'), htmlFor: 'live-contrast-slider', className: 'preview-filter-label' }), liveContrastSlider);
    const liveSaturationWrapper = createEl('span', { style: { display: 'inline-flex', flexDirection: 'column', gap: '2px', alignItems: 'center' } });
    liveSaturationWrapper.append(createEl('label', { textContent: t('saturation', 'Saturation'), htmlFor: 'live-saturation-slider', className: 'preview-filter-label' }), liveSaturationSlider);
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
    modalContent.append(modalTitle, statusEl, videoContainer, controlsContainer, liveFilterControls, closeButton);
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
