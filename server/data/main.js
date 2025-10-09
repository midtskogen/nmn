import * as uiManager from './ui_manager.js';
import * as mapHandler from './map_handler.js';
import * as chartHandler from './chart_handler.js';
import * as api from './api.js';
import { getSunTimes } from './calculations.js';
import { createEl, isHevcSupported } from './utils.js';

// --- Application State and I18n ---
let LANG = {};

/**
 * Translates a key into the current language, with placeholder support.
 * @param {string} key - The translation key from the language file.
 * @param {object} [replacements={}] - An object of placeholders to replace, e.g., {count: 5}.
 * @returns {string} The translated and formatted string.
 */
function t(key, replacements = {}) {
    let str = LANG[key] || key;
    for (const placeholder in replacements) {
        str = str.replace(`{${placeholder}}`, replacements[placeholder]);
    }
    return str;
}

/**
 * The main entry point for the application.
 * Fetches language data before initializing the rest of the application.
 */
async function main() {
    try {
        const response = await fetch('index.php?action=get_lang');
        if (!response.ok) throw new Error('Network response was not ok');
        LANG = await response.json();
    } catch (error) {
        console.error("Fatal: Could not load language file. UI text will be missing.", error);
        document.body.innerHTML = 'Error: Could not load application language data. Please try refreshing the page.';
        return;
    }
    
    // The main application logic starts once the DOM is fully loaded and language is fetched.
    initializeApp();
}

document.addEventListener('DOMContentLoaded', main);

/**
 * This function is called after the language file is loaded and the DOM is ready.
 */
function initializeApp() {
    // --- Application State ---
    let selectedStations = new Set();
    let stationsData = {}, cameraFovs = {};
    let passData = {}, aircraftData = {}, lightningData = [];
    let currentTaskId = null;
    let statusInterval = null;
    let isFirstCameraClickSincePassChange = true;
    let activeStationForSelection = null;
    let currentHighlightedPassId = null;
    let currentHighlightedCrossingId = null;

    // --- DOM Element Cache ---
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
        liveStreamControls: null
    };

    // --- Initialization ---
    
    // Dynamically create the placeholder for live stream controls and inject it into the DOM.
    const cameraFieldset = document.querySelector('fieldset.form-group legend').parentElement;
    dom.liveStreamControls = createEl('div', { id: 'live-stream-controls', style: 'display: none;' });
    cameraFieldset.insertAdjacentElement('afterend', dom.liveStreamControls);
    
    // Initialize all the major modules, passing the translation function.
    uiManager.initUIManager(dom, resetAll, t);
    chartHandler.initChart(t);
    mapHandler.initMap('map', handleMapMoveEnd, handleMapZoomEnd, t);

    // Set up all event listeners for user interaction.
    initEventListeners();
    
    // Fetch all the initial data needed to render the application.
    api.fetchInitialData()
        .then(data => {
            stationsData = data.stations;
            cameraFovs = data.fovs;
            lightningData = data.lightning;

            mapHandler.setMeteorData(data.meteors);
            mapHandler.setLightningData(data.lightning);

            Object.entries(stationsData).forEach(([stationId, station]) => {
                station.station.id = stationId;
                mapHandler.addStationMarker(stationId, station, handleStationClick);
            });

            if (data.kpData && !data.kpData.error) {
                const formattedKpData = formatKpData(data.kpData);
                const chartCtx = document.getElementById('aurora-chart').getContext('2d');
                chartHandler.plotAuroraChart(chartCtx, formattedKpData, handleChartBarClick);
            }

            uiManager.displayLightningStrikes(data.lightning, stationsData, cameraFovs, false, handleLightningSelect);
            mapHandler.displayLightningStrikes(false, handleLightningSelect);
            handleMapMoveEnd();
            uiManager.updateSelectedStationsUI(selectedStations, stationsData, startLiveStream);
        })
        .catch(error => {
            dom.formError.textContent = error.message;
        });
    uiManager.setUIState('ready');

    // --- Event Handlers & Callbacks ---

    function handleStationClick(e) {
        const clickedId = e.target.stationId;
        if (selectedStations.has(clickedId)) {
            selectedStations.delete(clickedId);
            mapHandler.updateStationMarkerIcon(clickedId, mapHandler.blueIcon);
        } else {
            selectedStations.add(clickedId);
            mapHandler.updateStationMarkerIcon(clickedId, mapHandler.redIcon);
        }

        if (clickedId === activeStationForSelection && !selectedStations.has(clickedId)) {
            uiManager.clearSelections(mapHandler);
            activeStationForSelection = null;
        }

        uiManager.updateSelectedStationsUI(selectedStations, stationsData, startLiveStream);
        handleMapMoveEnd();
    }

    function handleMapMoveEnd() {
        let lat, lon;
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

    function handleMapZoomEnd() {
        const isSatView = document.getElementById('satellite-toggle').checked;
        const isAircraftView = document.getElementById('aircraft-toggle').checked;
        const id = isSatView ? currentHighlightedPassId : currentHighlightedCrossingId;
        const type = isSatView ? 'satellite' : 'aircraft';
        mapHandler.highlightTrack(id, type, isSatView, isAircraftView);
        if (document.getElementById('meteor-toggle').checked) {
	    mapHandler.displayMeteors(handleMeteorClick, handleMeteorMouseover, handleMeteorMouseout);
	}

	// Redraw bearing lines if a pass/crossing and station are selected
	if (id && activeStationForSelection) {
	    const dataSource = isSatView ? passData.passes : aircraftData.crossings;
	    const idKey = isSatView ? 'pass_id' : 'crossing_id';
	    const item = dataSource?.find(p => p[idKey] === id);
	    
	    if (item) {
		const checkedCameras = [...document.querySelectorAll('input[name="cameras"]:checked')].map(cb => parseInt(cb.value, 10));
		const selectedCameraViews = item.camera_views.filter(cv =>
		    cv.station_id === activeStationForSelection && checkedCameras.includes(cv.camera)
		);
		
		if (selectedCameraViews.length > 0) {
		    mapHandler.drawBearingLines(item, selectedCameraViews, stationsData);
		}
	    }
        }
    }

    function handleChartBarClick(clickedDataPoint) {
        const clickedTimestamp = new Date(clickedDataPoint.timestamp);
        dom.dateInput.value = clickedTimestamp.toISOString().slice(0, 10);
        dom.hourSelect.value = clickedTimestamp.getUTCHours();
        dom.minuteSelect.value = clickedTimestamp.getUTCMinutes();
        dom.lengthSelect.value = 1;
        dom.intervalSelect.value = 1;
        dom.dateInput.dispatchEvent(new Event('change'));
    }

    function handlePassHeaderClick(id, type) {
        if ((type === 'satellite' && currentHighlightedPassId !== id) || (type === 'aircraft' && currentHighlightedCrossingId !== id)) {
            isFirstCameraClickSincePassChange = true;
        }
        currentHighlightedPassId = (type === 'satellite') ? id : null;
        currentHighlightedCrossingId = (type === 'aircraft') ? id : null;

        const isSatView = document.getElementById('satellite-toggle').checked;
        const isAircraftView = document.getElementById('aircraft-toggle').checked;
        mapHandler.highlightTrack(id, type, isSatView, isAircraftView);
        uiManager.highlightPassInPanel(id);
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

    function handleEventClick(pass, type, event) {
        handlePassHeaderClick(pass.pass_id || pass.crossing_id, type);
        const clickedStationId = event.station_id;
        const clickedCamera = event.camera;
        const camCheckbox = document.querySelector(`input[name="cameras"][value="${clickedCamera}"]`);
        if (isFirstCameraClickSincePassChange || activeStationForSelection !== clickedStationId) {
            document.querySelectorAll('input[name="cameras"]').forEach(cb => cb.checked = false);
            activeStationForSelection = clickedStationId;
            if (camCheckbox) camCheckbox.checked = true;
            isFirstCameraClickSincePassChange = false;
        } else if (camCheckbox) {
            camCheckbox.checked = !camCheckbox.checked;
        }

        selectedStations.clear();
        if (document.querySelectorAll('input[name="cameras"]:checked').length > 0) {
            selectedStations.add(activeStationForSelection);
        } else {
            activeStationForSelection = null;
            isFirstCameraClickSincePassChange = true;
        }

        Object.keys(mapHandler.getStationMarkers()).forEach(stationId => {
            mapHandler.updateStationMarkerIcon(stationId, selectedStations.has(stationId) ? mapHandler.redIcon : mapHandler.blueIcon);
        });
        uiManager.updateSelectedStationsUI(selectedStations, stationsData, startLiveStream);
        uiManager.updateFormFromSelection(dom, selectedStations, pass.pass_id || pass.crossing_id, pass, mapHandler, stationsData);
    }

    function handleLightningSelect(strike, shouldPan) {
        mapHandler.selectLightningStrikeOnMap(strike, shouldPan);
        uiManager.selectLightningStrikeInPanel(strike.id);
        const d = new Date(strike.time);
        dom.dateInput.value = d.toISOString().slice(0, 10);
        dom.hourSelect.value = d.getUTCHours();
        dom.minuteSelect.value = d.getUTCMinutes();
        dom.lengthSelect.value = 1;
        dom.intervalSelect.value = 1;
        dom.dateInput.dispatchEvent(new Event('change'));
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

    function handleMeteorClick(meteor) {
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

    function handleMeteorMouseover(meteor) {
        (meteor.station_ids || []).forEach(id => {
            const marker = mapHandler.getStationMarkers()[id];
            if (marker) marker.setIcon(mapHandler.yellowIcon);
        });
    }

    function handleMeteorMouseout(meteor) {
        (meteor.station_ids || []).forEach(id => {
            const marker = mapHandler.getStationMarkers()[id];
            if (marker) marker.setIcon(selectedStations.has(id) ? mapHandler.redIcon : mapHandler.blueIcon);
        });
    }

    // --- Utility & Business Logic Functions ---

    async function startLiveStream(stationId, cameraNum, resolution) {
        try {
            const streamTaskId = await api.startStream(stationId, cameraNum, resolution, isHevcSupported());
            uiManager.showVideoModal(stationId, cameraNum, resolution, streamTaskId);
        } catch (error) {
            alert(t('error_stream_start', { error: error.message }));
        }
    }

    function formatKpData(kpData) {
        const chartData = [];
        let lastDateLabel = '';
        kpData.slice(1).slice(-56).forEach(row => {
            const [timestamp, kpValueStr] = row;
            const kpValue = parseFloat(kpValueStr);
            if (timestamp && !isNaN(kpValue)) {
                const dateObj = new Date(timestamp.replace(' ', 'T') + 'Z');
                const currentDateLabel = dateObj.toISOString().slice(0, 10);
       
                const timeLabel = `${String(dateObj.getUTCHours()).padStart(2, '0')}:${String(dateObj.getUTCMinutes()).padStart(2, '0')}`;
                chartData.push({ label: (currentDateLabel !== lastDateLabel) ? currentDateLabel : timeLabel, value: kpValue, timestamp: dateObj.toISOString() });
                lastDateLabel = currentDateLabel;
         
            }
        });
        return chartData;
    }

    function startPassDownload(passId, type) {
        const dataSource = type === 'satellite' ? passData.passes : aircraftData.crossings;
        const idKey = type === 'satellite' ? 'pass_id' : 'crossing_id';
        const item = dataSource.find(p => p[idKey] === passId);
        if (!item) {
            alert(t('error_pass_not_found', { type: type }));
            return;
        }

        const primaryType = document.querySelector('input[name="primary_file_type"]:checked').value;
        const isHighRes = document.getElementById('high-resolution-switch').checked;
        const isLongInt = document.getElementById('long-integration-switch').checked;
        const fileType = primaryType === 'video'
            ? (isHighRes ? 'hires' : 'lowres')
            : (isHighRes ? (isLongInt ? 'image_long' : 'image') : (isLongInt ? 'image_lowres_long' : 'image_lowres'));
        const payload = {
            [type === 'satellite' ? 'pass_data' : 'crossing_data']: item,
            file_type: fileType,
            hevc_supported: isHevcSupported()
        };
        uiManager.setUIState('downloading');
        document.getElementById('results-panel')?.scrollIntoView({ behavior: 'smooth' });

        api.startDownload(payload).then(handleDownloadTask).catch(handleDownloadError);
    }

    function handleDownloadTask(taskId) {
        currentTaskId = taskId;
        statusInterval = api.pollDownloadStatus(taskId, {
            onProgress: (data) => {
                dom.progressBarInner.style.width = `${Math.round(data.step)}%`;
                dom.progressText.textContent = uiManager.translateMessage(data.message);
                uiManager.displayResults(data, dom, isHevcSupported());
            },
            onComplete: (data) => {
 
                uiManager.displayResults(data, dom, isHevcSupported());
                uiManager.setUIState('cooldown');
                if (currentTaskId) {
                    api.cleanupTask(currentTaskId);
                }
     
                currentTaskId = null;
            },
            onError: (data) => {
                dom.formError.textContent = `Error: ${uiManager.translateMessage(data.message)}`;
                uiManager.setUIState('ready');
                if (currentTaskId) api.cleanupTask(currentTaskId);
       
                currentTaskId = null;
            }
        });
    }

    function handleDownloadError(error) {
        dom.formError.textContent = uiManager.translateMessage(error.message);
        uiManager.setUIState('ready');
    }

    function resetAll() {
        uiManager.setDefaultFormValues();
        selectedStations.clear();
        Object.values(mapHandler.getStationMarkers()).forEach(marker => marker.setIcon(mapHandler.blueIcon));
        uiManager.updateSelectedStationsUI(selectedStations, stationsData, startLiveStream);

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

        dom.formError.textContent = '';
        dom.resultsLog.innerHTML = '';
        passData = {};
        aircraftData = {};
        uiManager.showPanelError('satellite', t('loading_passes'));
        uiManager.showPanelError('aircraft', t('loading_aircraft'));

        mapHandler.getMap().setView([64.7, 13.0], 5);
        uiManager.setUIState('ready');
    }

    // --- Event Listener Setup ---
    function initEventListeners() {
        document.getElementById('language-selector').addEventListener('click', (e) => {
            if (e.target.dataset.lang) {
                document.cookie = `lang=${e.target.dataset.lang};path=/;max-age=31536000`;
                location.reload();
            }
        });

        const timeChangeHandler = () => mapHandler.updateTimeDependentLayers(dom.dateInput.value, dom.hourSelect.value, dom.minuteSelect.value);
        dom.dateInput.addEventListener('change', timeChangeHandler);
        dom.hourSelect.addEventListener('change', timeChangeHandler);
        dom.minuteSelect.addEventListener('change', timeChangeHandler);

        document.getElementById('last-night-btn').addEventListener('click', () => {
            if (selectedStations.size === 0) return;
            const station = stationsData[selectedStations.values().next().value];
            const yesterday = new Date();
            yesterday.setUTCDate(yesterday.getUTCDate() - 1);
            const today = new Date();

            const yesterdayTimes = getSunTimes(yesterday, station.astronomy.latitude, station.astronomy.longitude, -6);
  
            const todayTimes = getSunTimes(today, station.astronomy.latitude, station.astronomy.longitude, -6);

            if (yesterdayTimes.type === 'polar_day') {
                dom.formError.textContent = t('error_last_night_polar_day');
                return;
            }
            
    
            let startTime = yesterdayTimes.type === 'polar_night'
                ? new Date(Date.UTC(yesterday.getUTCFullYear(), yesterday.getUTCMonth(), yesterday.getUTCDate(), 12, 0, 0))
                : yesterdayTimes.set;
            let endTime = todayTimes.type === 'polar_night'
                ? new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate(), 12, 0, 0))
                : todayTimes.rise;
            if (!startTime || !endTime) {
                dom.formError.textContent = t('error_last_night_suntimes');
                return;
            };

            dom.dateInput.value = startTime.toISOString().slice(0, 10);
            dom.hourSelect.value = startTime.getUTCHours();
            dom.minuteSelect.value = startTime.getUTCMinutes();
            dom.lengthSelect.value = 25;
            const durationMinutes = (endTime.getTime() - startTime.getTime()) / (1000 * 60);
            dom.intervalSelect.value = Math.max(1, Math.min(60, Math.round(durationMinutes / 25)));
            dom.dateInput.dispatchEvent(new Event('change'));
        });
        document.getElementById('now-button').addEventListener('click', () => {
            const now = new Date();
            now.setMinutes(now.getMinutes() - 2);
            dom.dateInput.value = now.toISOString().slice(0, 10);
            dom.hourSelect.value = now.getUTCHours();
            dom.minuteSelect.value = now.getUTCMinutes();
            dom.lengthSelect.value = 1;
            dom.intervalSelect.value = 1;
            dom.dateInput.dispatchEvent(new Event('change'));
        });
        dom.dateDisplayInput.addEventListener('click', () => { try { dom.dateInput.showPicker(); } catch (e) { dom.dateInput.click(); } });
        dom.dateInput.addEventListener('change', () => { if (dom.dateInput.value) dom.dateDisplayInput.value = dom.dateInput.value; });
        document.querySelector('input[name="primary_file_type"][value="video"]').addEventListener('change', () => { document.getElementById('long-integration-label').style.display = 'none'; });
        document.querySelector('input[name="primary_file_type"][value="image"]').addEventListener('change', () => { document.getElementById('long-integration-label').style.display = 'flex'; });
        
        document.getElementById('cloud-toggle').addEventListener('change', (e) => {
            if (e.target.checked) document.getElementById('aurora-toggle').checked = false;
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
        
        document.getElementById('satellite-toggle').addEventListener('change', (e) => {
            if (e.target.checked) {
                document.getElementById('aircraft-toggle').checked = false;
                uiManager.showPanel('satellite');
                uiManager.hidePanel('aircraft');
               
                mapHandler.highlightTrack(null, 'aircraft', false, false);
                api.fetchAllPasses({
                    onProgress: (data) => uiManager.updateTaskProgress('satellite', data),
                    onComplete: (data) => {
    
                        passData = data.data;
                        mapHandler.setPassData(data.data);
                        uiManager.displayAllPasses(data.data, { onHeaderClick: handlePassHeaderClick, onDownloadClick: (id) => startPassDownload(id, 'satellite'), onEventClick: (pass, event) => handleEventClick(pass, 'satellite', event) });
                        handleMapZoomEnd();
                    },
                    onError: (data) => uiManager.showPanelError('satellite', data.message)
                });
            } else {
                uiManager.hidePanel('satellite');
                mapHandler.highlightTrack(null, 'satellite', false, false);
            }
        });
        document.getElementById('aircraft-toggle').addEventListener('change', (e) => {
            if (e.target.checked) {
                document.getElementById('satellite-toggle').checked = false;
                uiManager.showPanel('aircraft');
                uiManager.hidePanel('satellite');
                mapHandler.highlightTrack(null, 'satellite', false, false);
          
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
                mapHandler.highlightTrack(null, 'aircraft', false, false);
            }
        });
        
        dom.downloadForm.addEventListener('submit', (event) => {
            event.preventDefault();

            const selectedCameras = [...document.querySelectorAll('input[name="cameras"]:checked')].map(cb => cb.value);
            if (selectedStations.size === 0) {
                dom.formError.textContent = t('error_no_station_selected');
          
                return;
            }
            if (selectedCameras.length === 0) {
                dom.formError.textContent = t('error_no_camera_selected');
                return;
            }
            dom.formError.textContent = '';

   
            const primaryType = document.querySelector('input[name="primary_file_type"]:checked').value;
            const isHighRes = document.getElementById('high-resolution-switch').checked;
            const isLongInt = document.getElementById('long-integration-switch').checked;
            const fileType = primaryType === 'video'
                ? (isHighRes ? 'hires' : 'lowres')
                : (isHighRes ? (isLongInt ? 'image_long' : 'image') : (isLongInt ? 'image_lowres_long' : 'image_lowres'));
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

        dom.cancelButton.addEventListener('click', () => {
            if (!currentTaskId) return;
            clearInterval(statusInterval);
            api.cancelTask(currentTaskId);
            dom.resultsLog.prepend(createEl('p', { innerHTML: `<strong>${t('download_cancelled')}</strong>`, className: 'error-msg' }));
            currentTaskId = null;
            dom.progressContainer.style.display = 'none';
            
            uiManager.setUIState('cooldown');
        });
    }
}
