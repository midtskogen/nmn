import { createEl, isHevcSupported } from './utils.js';
import { getSunTimes, calculateBearing } from './calculations.js';
import * as api from './api.js';

// --- Module-scoped variables ---
let dom = {}; // A cache for frequently accessed DOM elements.
let t = (key) => key; // The translation function, initialized to a fallback.
let hls = null; // Holds the HLS.js instance for playing HLS video streams.
let streamCountdownInterval = null; // Interval ID for the stream timeout countdown.
let activeStreamTaskId = null; // The task ID of the currently active stream.
let stopStreamTimeout = null; // Timeout ID to automatically close the modal.
let streamStatusPoller = null; // Interval ID for polling the stream's status.
let onFullscreenChange = null; // Holds the fullscreen change event handler.

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
        dom.stationList.innerHTML = [...selectedStations].map(stationId => `<li>${stationsData[stationId].station.code}</li>`).join('');
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
        const stationCode = stationsData[stationId]?.station?.code || 'station';
        const title = createEl('legend', { textContent: t('live_stream_title', { station_code: stationCode }) });
        const sdContainer = createEl('div', { className: 'live-stream-res-group' });
        const hdContainer = createEl('div', { className: 'live-stream-res-group' });
        
        for (let i = 1; i <= 7; i++) {
            const sdLink = createEl('span', { className: 'live-stream-link', textContent: `SD${i}`, onclick: () => onStreamLinkClick(stationId, i, 'lowres') });
            sdContainer.appendChild(sdLink);
            const hdLink = createEl('span', { className: 'live-stream-link', textContent: `HD${i}`, onclick: () => onStreamLinkClick(stationId, i, 'hires') });
            hdContainer.appendChild(hdLink);
        }
        dom.liveStreamControls.append(title, sdContainer, hdContainer);
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
        satelliteList.innerHTML = `<p style="color: #6c757d; margin: 0;">${t('no_visible_passes')}</p>`;
        return;
    }
    satelliteList.innerHTML = '';
    passData.passes.forEach((pass, index) => {
        const passDiv = createEl('div', { className: `satellite-group ${index % 2 === 0 ? 'pass-even' : 'pass-odd'}` });
        const earliestTime = new Date(pass.earliest_camera_utc);
        const formattedTimestamp = earliestTime.toISOString().slice(0, 19).replace('T', ' ');
        const headerHTML = t('pass_header', { satellite: pass.satellite, timestamp: formattedTimestamp }) +
                         ` <span class="magnitude">${t('pass_magnitude', { magnitude: pass.magnitude.toFixed(1) })}</span>`;
        const header = createEl('h6', { innerHTML: headerHTML, dataset: { passId: pass.pass_id }});
        header.addEventListener('click', () => onHeaderClick(pass.pass_id, 'satellite'));

        const downloadAllBtn = createEl('button', { textContent: t('download_all_button'), className: 'download-all-btn' });
        downloadAllBtn.onclick = (e) => { e.stopPropagation(); onHeaderClick(pass.pass_id, 'satellite'); onDownloadClick(pass.pass_id, 'satellite'); };

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
        aircraftList.innerHTML = `<p style="color: #6c757d; margin: 0;">${t('no_visible_aircraft')}</p>`;
        return;
    }
    aircraftList.innerHTML = '';
    aircraftData.crossings.forEach((crossing, index) => {
        const crossingDiv = createEl('div', { className: `satellite-group ${index % 2 === 0 ? 'pass-even' : 'pass-odd'}` });
        const earliestTime = new Date(crossing.earliest_camera_utc);
        const formattedTimestamp = earliestTime.toISOString().slice(0, 19).replace('T', ' ');
        const { callsign, origin, destination } = crossing.flight_info;
        const headerHTML = t('aircraft_header', { callsign: (callsign || '????'), origin: (origin || '?'), destination: (destination || '?'), timestamp: formattedTimestamp });
        const header = createEl('h6', { innerHTML: headerHTML, dataset: { crossingId: crossing.crossing_id }});
        header.addEventListener('click', () => onHeaderClick(crossing.crossing_id, 'aircraft'));

        const downloadAllBtn = createEl('button', { textContent: t('download_all_button'), className: 'download-all-btn' });
        downloadAllBtn.onclick = (e) => { e.stopPropagation(); onHeaderClick(crossing.crossing_id, 'aircraft'); onDownloadClick(crossing.crossing_id, 'aircraft'); };

        const headerContainer = createEl('div', { className: 'satellite-group-header' });
        headerContainer.append(header, downloadAllBtn);

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
export function displayLightningStrikes(strikes, stationInfo, cameraFovs, is24hFilter, onStrikeClick) {
    const lightningList = document.getElementById('lightning-list');
    document.querySelector('#lightning-panel h2').textContent = is24hFilter ? t('lightning_panel_title_24h') : t('lightning_panel_title');

    let filteredStrikes = strikes;
    if (is24hFilter) {
        const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
        filteredStrikes = strikes.filter(strike => new Date(strike.time) >= twentyFourHoursAgo);
    }

    if (!filteredStrikes || filteredStrikes.length === 0) {
        lightningList.innerHTML = `<p style="color: #6c757d; margin: 0;">${t('no_lightning_strikes')}</p>`;
        return;
    }

    lightningList.innerHTML = '';
    const ul = createEl('ul', { className: 'lightning-list' });
    filteredStrikes.sort((a, b) => new Date(b.time) - new Date(a.time)).forEach((strike, index) => {
        strike.id = `lightning-${index}`;
        const timestamp = new Date(strike.time).toISOString().slice(0, 19).replace('T', ' ');
        const nearestStation = Object.values(stationInfo).reduce((prev, curr) => L.latLng(strike.lat, strike.lon).distanceTo(L.latLng(prev.astronomy.latitude, prev.astronomy.longitude)) < L.latLng(strike.lat, strike.lon).distanceTo(L.latLng(curr.astronomy.latitude, curr.astronomy.longitude)) ? prev : curr);
     
        let stationText = '';
        if (nearestStation) {
            const inViewCams = getCamerasInView(nearestStation, strike, cameraFovs);
            const bearing = calculateBearing(nearestStation.astronomy.latitude, nearestStation.astronomy.longitude, strike.lat, strike.lon);
            const strikeTypeText = strike.type === 'cg' ? t('lightning_type_cg') : t('lightning_type_ic');
            const params = {
                station_code: nearestStation.station.code,
                dist: strike.dist.toFixed(1),
                bearing: Math.round(bearing),
                type: strikeTypeText
            };
            if (inViewCams.length > 0) {
                params.cams = inViewCams.join(', ');
                stationText = t('lightning_list_item_station_info', params);
            } else {
                stationText = t('lightning_list_item_station_info_no_cam', params);
            }
        }
        const li = createEl('li', { id: strike.id, innerHTML: `<span class="strike-type-indicator ${strike.type}">⚡</span> ${timestamp} ${stationText}` });
        li.onclick = () => onStrikeClick(strike, true);
        ul.appendChild(li);
    });
    lightningList.appendChild(ul);
}


/**
 * Renders the results of a completed download task in the results panel.
 * @param {object} resultData - The data object from the backend, containing files and errors.
 * @param {object} dom - The DOM element cache.
 * @param {boolean} hevcSupported - Whether the user's browser supports HEVC.
 */
export function displayResults(resultData, dom, hevcSupported) {
    dom.resultsLog.innerHTML = '';
    const stationResults = resultData.files || {};
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
                    const li = createEl('li');
                    const isVideo = file.url.endsWith('.mp4');
                    
                    if (file.thumb_url) {
                        const link = createEl('a', { href: file.url, target: '_blank', title: file.name });
                        const thumbContainer = createEl('div', { className: `thumbnail-container${isVideo ? ' video' : ''}` });
                        thumbContainer.appendChild(createEl('img', { src: file.thumb_url, alt: file.name, className: 'thumbnail-preview' }));
  
                        link.appendChild(thumbContainer);
                        li.appendChild(link);
                    } else {
                        li.appendChild(createEl('a', { href: file.url, target: '_blank', textContent: file.name, title: file.name }));
                    }

                    const linksContainer = createEl('div', { className: 'alternate-links' });
                    const allFilesForThisThumb = [{ url: file.url, name: file.name }, ...(file.alternatives || [])];
                    const preferredLinks = {};
                    
                    const getShortName = (filename) => {
                        if (filename.includes('_image_long_stacked.jpg')) return 'bhL';
                        if (filename.includes('_image_lowres_long_stacked.jpg')) return 'blL';
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
                        linksContainer.appendChild(createEl('a', { href: linkInfo.url, target: '_blank', textContent: shortName }));
                    });
                    if (linksContainer.hasChildNodes()) li.appendChild(linksContainer);
                    ul.appendChild(li);
                });
                timeSetDiv.appendChild(ul);
                dom.resultsLog.appendChild(timeSetDiv);
            });
        });
    }
    
    if (resultData.errors && resultData.errors.length > 0) {
        dom.resultsLog.appendChild(createEl('h4', { textContent: t('error_messages_title') }));
        const errorUl = createEl('ul');
        resultData.errors.forEach(error => errorUl.appendChild(createEl('li', { className: 'error-msg', textContent: translateMessage(error) })));
        dom.resultsLog.appendChild(errorUl);
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
export function showVideoModal(stationId, cameraNum, resolution, streamTaskId) {
    if (activeStreamTaskId) {
        hideVideoModal();
    }
    activeStreamTaskId = streamTaskId;
    
    const modalBackdrop = createEl('div', { id: 'video-modal-backdrop' });
    const modalContent = createEl('div', { id: 'video-modal-content' });
    const videoContainer = createEl('div', { id: 'video-container', style: { aspectRatio: resolution === 'lowres' ? '800 / 448' : '1920 / 1080' } });
    const videoEl = createEl('video', { id: 'live-video', muted: true, autoplay: true, playsinline: true });
    const gridOverlay = createEl('img', { id: 'grid-overlay-image' });
    const statusEl = createEl('p', { id: 'video-status', textContent: t('modal_starting_stream') });
    const controlsContainer = createEl('div', { className: 'video-controls-container' });
    const gridToggleContainer = createEl('div', { id: 'grid-toggle-container', style: 'display: none;' });
    const gridCheckbox = createEl('input', { type: 'checkbox', id: 'grid-overlay-toggle' });
    const gridLabel = createEl('label', { textContent: t('modal_grid_toggle'), htmlFor: 'grid-overlay-toggle' });
    gridToggleContainer.append(gridCheckbox, gridLabel);
    const fullscreenButton = createEl('button', { id: 'fullscreen-btn', textContent: t('modal_fullscreen_button') });
    controlsContainer.append(gridToggleContainer, fullscreenButton);
    const closeButton = createEl('button', { id: 'video-close-button', textContent: t('modal_close_button'), onclick: hideVideoModal });

    videoContainer.append(videoEl, gridOverlay);
    modalContent.append(statusEl, videoContainer, controlsContainer, closeButton);
    modalBackdrop.appendChild(modalContent);
    document.body.appendChild(modalBackdrop);
    
    videoEl.addEventListener('loadedmetadata', () => {
        if (videoEl.videoHeight > 0) videoContainer.style.aspectRatio = videoEl.videoWidth / videoEl.videoHeight;
    });
    gridCheckbox.addEventListener('change', () => { gridOverlay.style.opacity = gridCheckbox.checked ? '0.3' : '0'; });
    
    // Pan and Zoom logic
    let scale = 1, panX = 0, panY = 0, isPanning = false, startPanX = 0, startPanY = 0, panOriginX = 0, panOriginY = 0;
    const clamp = (val, min, max) => Math.min(Math.max(val, min), max);
    const updateTransform = () => {
        const transformValue = `translate(${panX}px, ${panY}px) scale(${scale})`;
        videoEl.style.transform = transformValue;
        gridOverlay.style.transform = transformValue;
    };
    videoEl.style.transformOrigin = '0 0';
    gridOverlay.style.transformOrigin = '0 0';
    
    const onWheel = e => {
        if (document.fullscreenElement) return;
        e.preventDefault();
        const rect = videoContainer.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const scaleAmount = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = clamp(scale * scaleAmount, 1, 8);
        const newPanX = mouseX - (mouseX - panX) * (newScale / scale);
        const newPanY = mouseY - (mouseY - panY) * (newScale / scale);
        scale = newScale;
        if (scale <= 1.01) {
            panX = 0;
            panY = 0;
        } else {
            const minPanX = -(videoContainer.clientWidth * (scale - 1));
            const minPanY = -(videoContainer.clientHeight * (scale - 1));
            panX = clamp(newPanX, minPanX, 0);
            panY = clamp(newPanY, minPanY, 0);
        }
        updateTransform();
        if (videoEl.paused) videoEl.play().catch(() => {});
    };
    
    const onMouseMove = e => {
        if (!isPanning) return;
        const newPanX = startPanX + (e.clientX - panOriginX);
        const newPanY = startPanY + (e.clientY - panOriginY);
        const minPanX = -(videoContainer.clientWidth * (scale - 1));
        const minPanY = -(videoContainer.clientHeight * (scale - 1));
        panX = clamp(newPanX, minPanX, 0);
        panY = clamp(newPanY, minPanY, 0);
        updateTransform();
    };
    
    const onMouseUp = () => {
        isPanning = false;
        videoContainer.style.cursor = 'grab';
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
    };

    const onMouseDown = e => {
        if (e.button !== 0 || document.fullscreenElement) return;
        e.preventDefault();
        isPanning = true;
        videoContainer.style.cursor = 'grabbing';
        panOriginX = e.clientX;
        panOriginY = e.clientY;
        startPanX = panX;
        startPanY = panY;
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
    };

    let cursorIdleTimer = null;
    const handleIdleCursor = () => {
        if (!videoContainer || !document.fullscreenElement) return;
        videoContainer.style.cursor = 'default';
        clearTimeout(cursorIdleTimer);
        cursorIdleTimer = setTimeout(() => { videoContainer.style.cursor = 'none'; }, 2000);
    };
    
    onFullscreenChange = () => {
        const isFullscreen = !!document.fullscreenElement;
        fullscreenButton.textContent = isFullscreen ? t('modal_exit_fullscreen_button') : t('modal_fullscreen_button');
        if (isFullscreen) {
            scale = 1; panX = 0; panY = 0;
            updateTransform();
            videoContainer.addEventListener('mousemove', handleIdleCursor);
            handleIdleCursor();
        } else {
            videoContainer.removeEventListener('mousemove', handleIdleCursor);
            clearTimeout(cursorIdleTimer);
            videoContainer.style.cursor = 'grab';
        }
    };

    fullscreenButton.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            videoContainer.requestFullscreen().catch(err => alert(t('modal_fullscreen_error', { error: err.message })));
        } else {
            document.exitFullscreen();
        }
    });
    
    videoContainer.addEventListener('wheel', onWheel);
    videoContainer.addEventListener('mousedown', onMouseDown);
    document.addEventListener('fullscreenchange', onFullscreenChange);

    streamStatusPoller = api.pollStreamStatus(streamTaskId, {
        onStatusUpdate: (data) => {
            if (statusEl) statusEl.textContent = translateMessage(data.message) || t('modal_status_updating');
        },
        onReady: (data) => {
            if (statusEl) statusEl.textContent = translateMessage(data.message) || t('modal_status_ready');
            const playlistUrl = `streams/${data.station_id}_${cameraNum}_${data.resolution}/playlist.m3u8`;

            if (Hls.isSupported()) {
                hls = new Hls({ maxBufferLength: 2, maxMaxBufferLength: 4, highBufferWatchdogPeriod: 2 });
                hls.loadSource(playlistUrl);
                hls.attachMedia(videoEl);
                hls.on(Hls.Events.MANIFEST_PARSED, () => videoEl.play());
            } else if (videoEl.canPlayType('application/vnd.apple.mpegurl')) {
                videoEl.src = playlistUrl;
                videoEl.addEventListener('canplay', () => videoEl.play());
            }

            const timeoutSeconds = data.timeout_seconds || 300;
            let timeLeft = timeoutSeconds;
            stopStreamTimeout = setTimeout(hideVideoModal, timeLeft * 1000);
            streamCountdownInterval = setInterval(() => {
                timeLeft--;
                const minutes = Math.floor(timeLeft / 60);
                const seconds = timeLeft % 60;
                const currentText = (statusEl.textContent || '').split(' | ')[0];
                statusEl.textContent = `${currentText} | ${t('modal_stream_stops_in', { minutes: minutes, seconds: String(seconds).padStart(2, '0') })}`;
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
            }
        })
        .catch(err => console.error("Could not fetch grid overlay:", err));
}

/**
 * Hides the video modal and cleans up all associated resources.
 */
function hideVideoModal() {
    if (streamCountdownInterval) clearInterval(streamCountdownInterval);
    if (streamStatusPoller) clearInterval(streamStatusPoller);
    if (stopStreamTimeout) clearTimeout(stopStreamTimeout);
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
            const isChecked = Array.from(checkedCameras).some(cb => parseInt(cb.value, 10) === camNum && stationId === currentStationId);
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

/** Hides a side panel. */
export function hidePanel(panelType) {
    document.getElementById(`${panelType}-panel-container`).style.display = 'none';
}

/** Displays an error message inside a panel's list area. */
export function showPanelError(panelType, message) {
    const listEl = document.getElementById(`${panelType}-list`);
    if (listEl) {
        listEl.innerHTML = `<p class="error-msg">${message}</p>`;
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
