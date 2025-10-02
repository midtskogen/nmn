import * as uiManager from './ui_manager.js';
import * as mapHandler from './map_handler.js';
import * as chartHandler from './chart_handler.js';
import * as api from './api.js';
import { getSunTimes } from './calculations.js';
import { createEl, isHevcSupported } from './utils.js';

// The main application logic starts once the DOM is fully loaded.
document.addEventListener('DOMContentLoaded', () => {
    // --- Application State ---
    // These variables hold the core state of the application, such as user selections,
    // fetched data, and the status of ongoing tasks.
    let selectedStations = new Set();
    let stationsData = {}, cameraFovs = {};
    let passData = {}, aircraftData = {}, lightningData = [];
    let currentTaskId = null; // Holds the ID of the active download/processing task.
    let statusInterval = null; // Holds the interval ID for polling task status.
    let isFirstCameraClickSincePassChange = true;
    let activeStationForSelection = null;
    let currentHighlightedPassId = null;
    let currentHighlightedCrossingId = null;

    // --- DOM Element Cache ---
    // Caching DOM elements improves performance by avoiding repeated `getElementById` calls.
    const dom = {
        map: document.getElementById('map'),
        downloadForm: document.getElementById('download-form'),
        downloadButton: document.getElementById('download-button'),
        cancelButton: document.getElementById('cancel-button'),
        resultsLog: document.getElementById('results-log'),
        formError: document.getElementById('form-error'),
        progressContainer: document.getElementById('progress-container'),
        progressBarInner: document.getElementById('progress-bar-inner'),
        progressText: document.getElementById('progress-text'),
        stationList: document.getElementById('station-list'),
        stationListPlaceholder: document.querySelector('#station-list-container p'),
        dateInput: document.getElementById('date'),
        dateDisplayInput: document.getElementById('date-display'),
        hourSelect: document.getElementById('hour'),
        minuteSelect: document.getElementById('minute'),
        lengthSelect: document.getElementById('length'),
        intervalSelect: document.getElementById('interval'),
        liveStreamControls: null // This element is created dynamically.
    };

    // --- Initialization ---
    /**
     * The main function that sets up the entire application.
     */
    function initializeApp() {
        // Dynamically create the placeholder for live stream controls and inject it into the DOM.
        const cameraFieldset = document.querySelector('fieldset.form-group legend').parentElement;
        dom.liveStreamControls = createEl('div', { id: 'live-stream-controls', style: 'display: none;' });
        cameraFieldset.insertAdjacentElement('afterend', dom.liveStreamControls);

        // Initialize all the major modules.
        uiManager.initUIManager(dom, resetAll); // Pass the resetAll function to the UI Manager.
        chartHandler.initChart();
        mapHandler.initMap('map', handleMapMoveEnd, handleMapZoomEnd);

        // Set up all event listeners for user interaction.
        initEventListeners();

        // Fetch all the initial data needed to render the application.
        api.fetchInitialData()
            .then(data => {
                // Store the fetched data in the application state.
                stationsData = data.stations;
                cameraFovs = data.fovs;
                lightningData = data.lightning;

                // Pass data to the relevant modules.
                mapHandler.setMeteorData(data.meteors);
                mapHandler.setLightningData(data.lightning);

                // Add a marker on the map for each station.
                Object.entries(stationsData).forEach(([stationId, station]) => {
                    station.station.id = stationId;
                    mapHandler.addStationMarker(stationId, station, handleStationClick);
                });

                // If Kp-index data is available, format it and plot the chart.
                if (data.kpData && !data.kpData.error) {
                    const formattedKpData = formatKpData(data.kpData);
                    const chartCtx = document.getElementById('aurora-chart').getContext('2d');
                    chartHandler.plotAuroraChart(chartCtx, formattedKpData, handleChartBarClick);
                }

                // Display initial lightning strike data.
                uiManager.displayLightningStrikes(data.lightning, stationsData, cameraFovs, false, handleLightningSelect);
                mapHandler.displayLightningStrikes(false, handleLightningSelect);
                handleMapMoveEnd(); // Trigger an initial chart background update based on the map center.

                uiManager.updateSelectedStationsUI(selectedStations, stationsData, startLiveStream);
            })
            .catch(error => {
                dom.formError.textContent = error.message;
            });

        uiManager.setUIState('ready'); // Set the initial UI state.
    }

    // --- Event Handlers & Callbacks ---

    /**
     * Handles clicks on station markers on the map to select or deselect them.
     * @param {Event} e - The Leaflet map click event.
     */
    function handleStationClick(e) {
        const clickedId = e.target.stationId;
        // Toggle station selection.
        if (selectedStations.has(clickedId)) {
            selectedStations.delete(clickedId);
            mapHandler.updateStationMarkerIcon(clickedId, mapHandler.blueIcon);
        } else {
            selectedStations.add(clickedId);
            mapHandler.updateStationMarkerIcon(clickedId, mapHandler.redIcon);
        }

        // If a station that was active for camera selection is deselected, clear the selections.
        if (clickedId === activeStationForSelection && !selectedStations.has(clickedId)) {
            uiManager.clearSelections(mapHandler);
            activeStationForSelection = null;
        }

        uiManager.updateSelectedStationsUI(selectedStations, stationsData, startLiveStream);
        handleMapMoveEnd(); // Update chart background based on the new primary station.
    }

    /**
     * Updates the day/night background of the Kp-index chart when the map is moved or a station is selected.
     */
    function handleMapMoveEnd() {
        let lat, lon;
        // Use the location of the first selected station, or the map center if no station is selected.
        if (selectedStations.size > 0) {
            const station = stationsData[selectedStations.values().next().value];
            lat = station.astronomy.latitude;
            lon = station.astronomy.longitude;
        } else {
            const center = mapHandler.getMap().getCenter();
            lat = center.lat;
            lon = center.lng;
        }
        chartHandler.updateChartBackground(lat, lon);
    }

    /**
     * Handles events after the map zoom level changes, primarily for redrawing map features
     * that are zoom-dependent, like track labels and meteor shapes.
     */
    function handleMapZoomEnd() {
        const isSatView = document.getElementById('satellite-toggle').checked;
        const isAircraftView = document.getElementById('aircraft-toggle').checked;
        const id = isSatView ? currentHighlightedPassId : currentHighlightedCrossingId;
        const type = isSatView ? 'satellite' : 'aircraft';
        // Re-highlight the current track to adjust label density for the new zoom level.
        mapHandler.highlightTrack(id, type, isSatView, isAircraftView);

        // Redraw meteors if the layer is active.
        if (document.getElementById('meteor-toggle').checked) {
            mapHandler.displayMeteors(handleMeteorClick, handleMeteorMouseover, handleMeteorMouseout);
        }
    }

    /**
     * Handles clicks on the bars of the Kp-index chart, populating the download form
     * with the corresponding date and time.
     * @param {object} clickedDataPoint - The data object for the clicked bar.
     */
    function handleChartBarClick(clickedDataPoint) {
        const clickedTimestamp = new Date(clickedDataPoint.timestamp);
        dom.dateInput.value = clickedTimestamp.toISOString().slice(0, 10);
        dom.hourSelect.value = clickedTimestamp.getUTCHours();
        dom.minuteSelect.value = clickedTimestamp.getUTCMinutes();
        dom.lengthSelect.value = 1;
        dom.intervalSelect.value = 1;
        dom.dateInput.dispatchEvent(new Event('change')); // Trigger change event to update map layers.
    }

    /**
     * Handles clicks on the header of a satellite pass or aircraft crossing in the side panel.
     * @param {string} id - The ID of the pass or crossing.
     * @param {string} type - 'satellite' or 'aircraft'.
     */
    function handlePassHeaderClick(id, type) {
        // Reset state if a different pass is selected.
        if ((type === 'satellite' && currentHighlightedPassId !== id) || (type === 'aircraft' && currentHighlightedCrossingId !== id)) {
            isFirstCameraClickSincePassChange = true;
        }
        currentHighlightedPassId = (type === 'satellite') ? id : null;
        currentHighlightedCrossingId = (type === 'aircraft') ? id : null;

        const isSatView = document.getElementById('satellite-toggle').checked;
        const isAircraftView = document.getElementById('aircraft-toggle').checked;
        mapHandler.highlightTrack(id, type, isSatView, isAircraftView); // Highlight the track on the map.
        uiManager.highlightPassInPanel(id); // Highlight the item in the panel.

        // Find the corresponding data and populate the form with the pass's start time.
        const data_source = (type === 'satellite') ? passData.passes : aircraftData.crossings;
        const id_key = (type === 'satellite') ? 'pass_id' : 'crossing_id';
        const item = data_source?.find(p => p[id_key] === id);
        if (item) {
            const earliestTime = new Date(item.earliest_camera_utc);
            dom.dateInput.value = earliestTime.toISOString().slice(0, 10);
            dom.hourSelect.value = earliestTime.getUTCHours();
            dom.minuteSelect.value = earliestTime.getUTCMinutes();
            dom.dateInput.dispatchEvent(new Event('change'));
        }
    }

    /**
     * Handles clicks on individual camera view events within a pass/crossing.
     * This logic manages the complex state of selecting a station and specific cameras
     * related to that event.
     * @param {object} pass - The pass/crossing data object.
     * @param {string} type - 'satellite' or 'aircraft'.
     * @param {object} event - The specific camera view event that was clicked.
     */
    function handleEventClick(pass, type, event) {
        handlePassHeaderClick(pass.pass_id || pass.crossing_id, type);
        const clickedStationId = event.station_id;
        const clickedCamera = event.camera;
        const camCheckbox = document.querySelector(`input[name="cameras"][value="${clickedCamera}"]`);

        // If this is the first camera clicked for this pass, or a different station is chosen,
        // clear all previous camera selections and select only the clicked one.
        if (isFirstCameraClickSincePassChange || activeStationForSelection !== clickedStationId) {
            document.querySelectorAll('input[name="cameras"]').forEach(cb => cb.checked = false);
            activeStationForSelection = clickedStationId;
            if (camCheckbox) camCheckbox.checked = true;
            isFirstCameraClickSincePassChange = false;
        } else if (camCheckbox) {
            // Otherwise, just toggle the clicked camera's checkbox.
            camCheckbox.checked = !camCheckbox.checked;
        }

        // Update the master set of selected stations based on the active station and checked cameras.
        selectedStations.clear();
        if (document.querySelectorAll('input[name="cameras"]:checked').length > 0) {
            selectedStations.add(activeStationForSelection);
        } else {
            activeStationForSelection = null;
            isFirstCameraClickSincePassChange = true;
        }

        // Update map markers and the UI to reflect the new selection.
        Object.keys(mapHandler.getStationMarkers()).forEach(stationId => {
            mapHandler.updateStationMarkerIcon(stationId, selectedStations.has(stationId) ? mapHandler.redIcon : mapHandler.blueIcon);
        });
        uiManager.updateSelectedStationsUI(selectedStations, stationsData, startLiveStream);
        uiManager.updateFormFromSelection(dom, selectedStations, pass.pass_id || pass.crossing_id, pass, mapHandler, stationsData);
    }

    /**
     * Handles selecting a lightning strike, either from the map or the side panel.
     * @param {object} strike - The lightning strike data object.
     * @param {boolean} shouldPan - Whether the map should pan to the strike's location.
     */
    function handleLightningSelect(strike, shouldPan) {
        mapHandler.selectLightningStrikeOnMap(strike.id, shouldPan);
        uiManager.selectLightningStrikeInPanel(strike.id);

        // Set the form time to the time of the strike.
        const d = new Date(strike.time);
        dom.dateInput.value = d.toISOString().slice(0, 10);
        dom.hourSelect.value = d.getUTCHours();
        dom.minuteSelect.value = d.getUTCMinutes();
        dom.lengthSelect.value = 1;
        dom.intervalSelect.value = 1;
        dom.dateInput.dispatchEvent(new Event('change'));

        // Automatically select the nearest station and the cameras that would have seen the strike.
        const nearestStation = Object.values(stationsData).reduce((prev, curr) => L.latLng(strike.lat, strike.lon).distanceTo(L.latLng(prev.astronomy.latitude, prev.astronomy.longitude)) < L.latLng(strike.lat, strike.lon).distanceTo(L.latLng(curr.astronomy.latitude, curr.astronomy.longitude)) ? prev : curr);
        if (nearestStation) {
            selectedStations.clear();
            selectedStations.add(nearestStation.station.id);
            Object.values(mapHandler.getStationMarkers()).forEach(m => m.setIcon(selectedStations.has(m.stationId) ? mapHandler.redIcon : mapHandler.blueIcon));
            uiManager.updateSelectedStationsUI(selectedStations, stationsData, startLiveStream);
            const inViewCams = uiManager.getCamerasInView(nearestStation, strike, cameraFovs);
            document.querySelectorAll('input[name="cameras"]').forEach(cb => cb.checked = inViewCams.includes(cb.value));
        }
    }

    /**
     * Handles clicking on a meteor track on the map.
     * @param {object} meteor - The meteor data object.
     */
    function handleMeteorClick(meteor) {
        // Set form time and select the observing stations.
        const d = new Date(meteor.timestamp);
        dom.dateInput.value = d.toISOString().slice(0, 10);
        dom.hourSelect.value = d.getUTCHours();
        dom.minuteSelect.value = d.getUTCMinutes();
        dom.lengthSelect.value = 1;
        dom.intervalSelect.value = 1;
        dom.dateInput.dispatchEvent(new Event('change'));
        selectedStations.clear();
        (meteor.station_ids || []).forEach(id => selectedStations.add(id));
        Object.entries(mapHandler.getStationMarkers()).forEach(([id, marker]) => marker.setIcon(selectedStations.has(id) ? mapHandler.redIcon : mapHandler.blueIcon));
        uiManager.updateSelectedStationsUI(selectedStations, stationsData, startLiveStream);
    }

    /**
     * Handles mouseover on a meteor, highlighting the participating stations.
     * @param {object} meteor - The meteor data object.
     */
    function handleMeteorMouseover(meteor) {
        (meteor.station_ids || []).forEach(id => {
            const marker = mapHandler.getStationMarkers()[id];
            if (marker) marker.setIcon(mapHandler.yellowIcon);
        });
    }

    /**
     * Handles mouseout from a meteor, resetting station marker colors.
     * @param {object} meteor - The meteor data object.
     */
    function handleMeteorMouseout(meteor) {
        (meteor.station_ids || []).forEach(id => {
            const marker = mapHandler.getStationMarkers()[id];
            if (marker) marker.setIcon(selectedStations.has(id) ? mapHandler.redIcon : mapHandler.blueIcon);
        });
    }

    // --- Utility & Business Logic Functions ---

    /**
     * Initiates a live video stream for a given station and camera.
     * @param {string} stationId
     * @param {number} cameraNum
     * @param {string} resolution
     */
    async function startLiveStream(stationId, cameraNum, resolution) {
        try {
            const streamTaskId = await api.startStream(stationId, cameraNum, resolution, isHevcSupported());
            uiManager.showVideoModal(stationId, cameraNum, resolution, streamTaskId);
        } catch (error) {
            alert(`Kunne ikke starte videostrøm: ${error.message}`);
        }
    }

    /**
     * Formats the raw Kp-index data from the API into a structure usable by Chart.js.
     * @param {Array} kpData - Raw Kp-data array from the API.
     * @returns {Array<object>} Formatted data for the chart.
     */
    function formatKpData(kpData) {
        const chartData = [];
        let lastDateLabel = '';
        // Slice to get the last 56 entries (approx 7 days of 3-hour intervals).
        kpData.slice(1).slice(-56).forEach(row => {
            const [timestamp, kpValueStr] = row;
            const kpValue = parseFloat(kpValueStr);
            if (timestamp && !isNaN(kpValue)) {
                const dateObj = new Date(timestamp.replace(' ', 'T') + 'Z');
                const currentDateLabel = dateObj.toISOString().slice(0, 10);
                const timeLabel = `${String(dateObj.getUTCHours()).padStart(2, '0')}:${String(dateObj.getUTCMinutes()).padStart(2, '0')}`;
                // To reduce clutter, only show the date for the first entry of that day.
                chartData.push({ label: (currentDateLabel !== lastDateLabel) ? currentDateLabel : timeLabel, value: kpValue, timestamp: dateObj.toISOString() });
                lastDateLabel = currentDateLabel;
            }
        });
        return chartData;
    }

    /**
     * Starts the download process for a specific pre-calculated pass or crossing.
     * @param {string} passId - The ID of the pass/crossing.
     * @param {string} type - 'satellite' or 'aircraft'.
     */
    function startPassDownload(passId, type) {
        const dataSource = type === 'satellite' ? passData.passes : aircraftData.crossings;
        const idKey = type === 'satellite' ? 'pass_id' : 'crossing_id';
        const item = dataSource.find(p => p[idKey] === passId);
        if (!item) {
            alert(`Kunne ikke finne data for ${type}.`);
            return;
        }

        // Construct the file type string from the form controls.
        const primaryType = document.querySelector('input[name="primary_file_type"]:checked').value;
        const isHighRes = document.getElementById('high-resolution-switch').checked;
        const isLongInt = document.getElementById('long-integration-switch').checked;
        const fileType = primaryType === 'video'
            ? (isHighRes ? 'hires' : 'lowres')
            : (isHighRes ? (isLongInt ? 'image_long' : 'image') : (isLongInt ? 'image_lowres_long' : 'image_lowres'));

        // Prepare the payload and start the download.
        const payload = {
            [type === 'satellite' ? 'pass_data' : 'crossing_data']: item,
            file_type: fileType,
            hevc_supported: isHevcSupported()
        };
        uiManager.setUIState('downloading');
        document.getElementById('results-panel')?.scrollIntoView({ behavior: 'smooth' });

        api.startDownload(payload).then(handleDownloadTask).catch(handleDownloadError);
    }

    /**
     * Handles the successful initiation of a download task by starting the polling process.
     * @param {string} taskId - The new task ID.
     */
    function handleDownloadTask(taskId) {
        currentTaskId = taskId;
        statusInterval = api.pollDownloadStatus(taskId, {
            onProgress: (data) => {
                dom.progressBarInner.style.width = `${Math.round(data.step)}%`;
                dom.progressText.textContent = data.message;
                uiManager.displayResults(data, dom, isHevcSupported());
            },
            onComplete: (data) => {
                uiManager.displayResults(data, dom, isHevcSupported());
                uiManager.setUIState('cooldown');
                if (currentTaskId) {
                    api.cleanupTask(currentTaskId); // Clean up temporary files on the server.
                }
                currentTaskId = null;
            },
            onError: (data) => {
                dom.formError.textContent = `Feil: ${data.message}`;
                uiManager.setUIState('ready');
                if (currentTaskId) api.cleanupTask(currentTaskId);
                currentTaskId = null;
            }
        });
    }

    /**
     * Handles errors that occur during the *initiation* of a download task.
     * @param {Error} error
     */
    function handleDownloadError(error) {
        dom.formError.textContent = error.message;
        uiManager.setUIState('ready');
    }

    /**
     * Resets the entire application UI and state to its default initial state.
     */
    function resetAll() {
        uiManager.setDefaultFormValues();
        selectedStations.clear();
        Object.values(mapHandler.getStationMarkers()).forEach(marker => marker.setIcon(mapHandler.blueIcon));
        uiManager.updateSelectedStationsUI(selectedStations, stationsData, startLiveStream);

        // Turn off all toggleable layers and hide their corresponding panels.
        ['cloud', 'aurora', 'terminator', 'satellite', 'aircraft', 'lightning', 'meteor'].forEach(type => {
            const toggle = document.getElementById(`${type}-toggle`);
            if (toggle && toggle.checked) {
                toggle.checked = false;
                toggle.dispatchEvent(new Event('change'));
            }
        });
        if(document.getElementById('lightning-24h-toggle').checked) {
            document.getElementById('lightning-24h-toggle').checked = false;
            document.getElementById('lightning-24h-toggle').dispatchEvent(new Event('change'));
        }

        // Clear results and reset data.
        dom.formError.textContent = '';
        dom.resultsLog.innerHTML = '';
        passData = {};
        aircraftData = {};
        uiManager.showPanelError('satellite', 'Velg "Vis satelittpasseringer" for å laste data.');
        uiManager.showPanelError('aircraft', 'Velg "Vis flytrafikk" for å laste data.');

        mapHandler.getMap().setView([64.7, 13.0], 5); // Reset map view.
        uiManager.setUIState('ready');
    }

    // --- Event Listener Setup ---
    /**
     * Binds all necessary event listeners to the DOM elements.
     */
    function initEventListeners() {
        // Handler for any time-related form input changes.
        const timeChangeHandler = () => mapHandler.updateTimeDependentLayers(dom.dateInput.value, dom.hourSelect.value, dom.minuteSelect.value);
        dom.dateInput.addEventListener('change', timeChangeHandler);
        dom.hourSelect.addEventListener('change', timeChangeHandler);
        dom.minuteSelect.addEventListener('change', timeChangeHandler);

        // "Siste natt" button handler.
        document.getElementById('last-night-btn').addEventListener('click', () => {
            if (selectedStations.size === 0) return;
            const station = stationsData[selectedStations.values().next().value];
            const yesterday = new Date();
            yesterday.setUTCDate(yesterday.getUTCDate() - 1);
            const today = new Date();

            const yesterdayTimes = getSunTimes(yesterday, station.astronomy.latitude, station.astronomy.longitude, -6);
            const todayTimes = getSunTimes(today, station.astronomy.latitude, station.astronomy.longitude, -6);

            if (yesterdayTimes.type === 'polar_day') {
                dom.formError.textContent = 'Kan ikke beregne "siste natt" under midnattssol.';
                return;
            }
            
            // Determine the start (sunset yesterday) and end (sunrise today) of the night period.
            let startTime = yesterdayTimes.type === 'polar_night'
                ? new Date(Date.UTC(yesterday.getUTCFullYear(), yesterday.getUTCMonth(), yesterday.getUTCDate(), 12, 0, 0))
                : yesterdayTimes.set;
            let endTime = todayTimes.type === 'polar_night'
                ? new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate(), 12, 0, 0))
                : todayTimes.rise;

            if (!startTime || !endTime) {
                dom.formError.textContent = 'Kunne ikke finne soloppgang/-nedgang for å bestemme natt.';
                return;
            };

            // Set form values to span this night period.
            dom.dateInput.value = startTime.toISOString().slice(0, 10);
            dom.hourSelect.value = startTime.getUTCHours();
            dom.minuteSelect.value = startTime.getUTCMinutes();
            dom.lengthSelect.value = 25; // Default number of intervals.
            const durationMinutes = (endTime.getTime() - startTime.getTime()) / (1000 * 60);
            dom.intervalSelect.value = Math.max(1, Math.min(60, Math.round(durationMinutes / 25)));
            dom.dateInput.dispatchEvent(new Event('change'));
        });

        // "Nå" button handler.
        document.getElementById('now-button').addEventListener('click', () => {
            const now = new Date();
            now.setMinutes(now.getMinutes() - 2); // Set to 2 minutes ago.
            dom.dateInput.value = now.toISOString().slice(0, 10);
            dom.hourSelect.value = now.getUTCHours();
            dom.minuteSelect.value = now.getUTCMinutes();
            dom.lengthSelect.value = 1;
            dom.intervalSelect.value = 1;
            dom.dateInput.dispatchEvent(new Event('change'));
        });

        // Custom date input handling to show a native picker on all devices.
        dom.dateDisplayInput.addEventListener('click', () => { try { dom.dateInput.showPicker(); } catch (e) { dom.dateInput.click(); } });
        dom.dateInput.addEventListener('change', () => { if (dom.dateInput.value) dom.dateDisplayInput.value = dom.dateInput.value; });

        // Toggle visibility of the "long integration" switch based on file type selection.
        document.querySelector('input[name="primary_file_type"][value="video"]').addEventListener('change', () => { document.getElementById('long-integration-label').style.display = 'none'; });
        document.querySelector('input[name="primary_file_type"][value="image"]').addEventListener('change', () => { document.getElementById('long-integration-label').style.display = 'flex'; });

        // --- Map Layer Toggles ---
        document.getElementById('cloud-toggle').addEventListener('change', (e) => {
            if (e.target.checked) document.getElementById('aurora-toggle').checked = false; // Cloud and Aurora are mutually exclusive.
            mapHandler.toggleLayer('cloud', e.target.checked, dom.dateInput.value);
            mapHandler.toggleLayer('aurora', false);
        });
        document.getElementById('aurora-toggle').addEventListener('change', (e) => {
            if (e.target.checked) document.getElementById('cloud-toggle').checked = false;
            mapHandler.toggleLayer('aurora', e.target.checked, dom.dateInput.value);
            mapHandler.toggleLayer('cloud', false);
        });
        document.getElementById('terminator-toggle').addEventListener('change', (e) => mapHandler.toggleLayer('terminator', e.target.checked, dom.dateInput.value, dom.hourSelect.value, dom.minuteSelect.value));
        document.getElementById('lightning-toggle').addEventListener('change', (e) => uiManager.togglePanelAndLayer('lightning', e.target.checked, mapHandler));
        document.getElementById('meteor-toggle').addEventListener('change', (e) => mapHandler.toggleMeteorLayer(e.target.checked, handleMeteorClick, handleMeteorMouseover, handleMeteorMouseout));
        document.getElementById('lightning-24h-toggle').addEventListener('change', () => {
            const is24h = document.getElementById('lightning-24h-toggle').checked;
            uiManager.displayLightningStrikes(lightningData, stationsData, cameraFovs, is24h, handleLightningSelect);
            mapHandler.displayLightningStrikes(is24h, handleLightningSelect);
        });

        // --- Satellite and Aircraft Panel Toggles ---
        document.getElementById('satellite-toggle').addEventListener('change', (e) => {
            if (e.target.checked) {
                document.getElementById('aircraft-toggle').checked = false; // Mutually exclusive.
                uiManager.showPanel('satellite');
                uiManager.hidePanel('aircraft');
                mapHandler.highlightTrack(null, 'aircraft', false, false); // Clear old tracks.
                // Fetch satellite pass data when the toggle is enabled.
                api.fetchAllPasses({
                    onProgress: (data) => uiManager.updateTaskProgress('satellite', data),
                    onComplete: (data) => {
                        passData = data.data;
                        mapHandler.setPassData(data.data);
                        uiManager.displayAllPasses(data.data, { onHeaderClick: handlePassHeaderClick, onDownloadClick: (id) => startPassDownload(id, 'satellite'), onEventClick: (pass, event) => handleEventClick(pass, 'satellite', event) });
                        handleMapZoomEnd(); // Redraw tracks if needed.
                    },
                    onError: (data) => uiManager.showPanelError('satellite', data.message)
                });
            } else {
                uiManager.hidePanel('satellite');
                mapHandler.highlightTrack(null, 'satellite', false, false); // Clear tracks.
            }
        });
        document.getElementById('aircraft-toggle').addEventListener('change', (e) => {
            if (e.target.checked) {
                document.getElementById('satellite-toggle').checked = false;
                uiManager.showPanel('aircraft');
                uiManager.hidePanel('satellite');
                mapHandler.highlightTrack(null, 'satellite', false, false); // Clear old tracks.
                // Fetch aircraft crossing data when the toggle is enabled.
                api.fetchAllAircraftCrossings({
                    onProgress: (data) => uiManager.updateTaskProgress('aircraft', data),
                    onComplete: (data) => {
                        aircraftData = data.data;
                        mapHandler.setAircraftData(data.data);
                        uiManager.displayAllAircraft(data.data, { onHeaderClick: handlePassHeaderClick, onDownloadClick: (id) => startPassDownload(id, 'aircraft'), onEventClick: (pass, event) => handleEventClick(pass, 'aircraft', event) });
                        handleMapZoomEnd();
                    },
                    onError: (data) => uiManager.showPanelError('aircraft', data.message)
                });
            } else {
                uiManager.hidePanel('aircraft');
                mapHandler.highlightTrack(null, 'aircraft', false, false); // Clear tracks.
            }
        });

        // Main download form submission handler.
        dom.downloadForm.addEventListener('submit', (event) => {
            event.preventDefault();

            // Validate form inputs.
            const selectedCameras = [...document.querySelectorAll('input[name="cameras"]:checked')].map(cb => cb.value);
            if (selectedStations.size === 0) {
                dom.formError.textContent = 'Velg minst én stasjon.';
                return;
            }
            if (selectedCameras.length === 0) {
                dom.formError.textContent = 'Velg minst ett kamera.';
                return;
            }
            dom.formError.textContent = '';

            // Construct the file type string.
            const primaryType = document.querySelector('input[name="primary_file_type"]:checked').value;
            const isHighRes = document.getElementById('high-resolution-switch').checked;
            const isLongInt = document.getElementById('long-integration-switch').checked;
            const fileType = primaryType === 'video'
                ? (isHighRes ? 'hires' : 'lowres')
                : (isHighRes ? (isLongInt ? 'image_long' : 'image') : (isLongInt ? 'image_lowres_long' : 'image_lowres'));

            // Build the JSON payload for the download request.
            const payload = {
                stations: [...selectedStations],
                date: dom.dateInput.value,
                hour: dom.hourSelect.value,
                minute: dom.minuteSelect.value,
                length: dom.lengthSelect.value,
                interval: dom.intervalSelect.value,
                cameras: selectedCameras,
                file_type: fileType,
                hevc_supported: isHevcSupported()
            };

            // If a satellite pass or aircraft crossing is currently highlighted,
            // include its data in the payload. This tells the backend to download
            // files specifically for that event, with track overlays.
            if (currentHighlightedPassId && passData.passes) {
                const pass = passData.passes.find(p => p.pass_id === currentHighlightedPassId);
                if (pass) {
                    payload.pass_data = pass;
                }
            } else if (currentHighlightedCrossingId && aircraftData.crossings) {
                const crossing = aircraftData.crossings.find(c => c.crossing_id === currentHighlightedCrossingId);
                if (crossing) {
                    payload.crossing_data = crossing;
                }
            }

            uiManager.setUIState('downloading');
            document.getElementById('results-panel')?.scrollIntoView({ behavior: 'smooth' });
            api.startDownload(payload).then(handleDownloadTask).catch(handleDownloadError);
        });

        // Cancel button handler.
        dom.cancelButton.addEventListener('click', () => {
            if (!currentTaskId) return;
            clearInterval(statusInterval);
            api.cancelTask(currentTaskId);
            dom.resultsLog.prepend(createEl('p', { innerHTML: '<strong>Nedlasting avbrutt.</strong>', className: 'error-msg' }));
            currentTaskId = null;
            dom.progressContainer.style.display = 'none';
            uiManager.setUIState('cooldown');
        });
    }

    // --- Start the application ---
    initializeApp();
});
