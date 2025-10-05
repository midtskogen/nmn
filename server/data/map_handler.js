import { createEl } from './utils.js';

// --- Module-scoped variables ---
let map;
let t = (key) => key; // Translation function fallback
let passData = {}, aircraftData = {}, meteorData = [], lightningData = [];
let stationMarkers = {};

// Layer Groups for managing map features
const layerGroups = {
    stations: L.layerGroup(),
    groundTracks: L.layerGroup(),
    skyTracks: L.layerGroup(),
    meteors: L.layerGroup(),
    lightning: L.layerGroup(),
    bearingLines: L.layerGroup(),
    timeLayers: { // Layers that depend on the selected time
        terminator: null,
        cloud: null,
        aurora: null
    }
};

// --- Icon Definitions ---
const createIcon = (color) => new L.Icon({
    iconUrl: `https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-${color}.png`,
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

export const blueIcon = createIcon('blue');
export const redIcon = createIcon('red');
export const yellowIcon = createIcon('yellow');

// --- Initialization ---

/**
 * Initializes the Leaflet map, adds tile layers, controls, and event listeners.
 * @param {string} mapId - The ID of the div element for the map.
 * @param {function} onMoveEnd - Callback for when the map finishes moving.
 * @param {function} onZoomEnd - Callback for when the map finishes zooming.
 * @param {function} translationFunc - The main translation function.
 */
export function initMap(mapId, onMoveEnd, onZoomEnd, translationFunc) {
    t = translationFunc;
    map = L.map(mapId, {
        center: [64.7, 13.0],
        zoom: 5,
        worldCopyJump: true
    });

    const baseLayers = {
        "Street": L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map),
        "Satellite": L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri'
        })
    };

    L.control.layers(baseLayers).addTo(map);
    Object.values(layerGroups).forEach(lg => {
        if (lg && typeof lg.addTo === 'function') lg.addTo(map);
    });

    map.on('moveend', onMoveEnd);
    map.on('zoomend', onZoomEnd);
}

// --- Data Setters ---
export const setPassData = (data) => { passData = data; };
export const setAircraftData = (data) => { aircraftData = data; };
export const setMeteorData = (data) => { meteorData = data; };
export const setLightningData = (data) => { lightningData = data; };

// --- Getters ---
export const getMap = () => map;
export const getStationMarkers = () => stationMarkers;

// --- Feature Drawing & Management ---

/**
 * Adds a station marker to the map.
 * @param {string} id - The station ID.
 * @param {object} station - The station data object.
 * @param {function} onClick - The click event handler for the marker.
 */
export function addStationMarker(id, station, onClick) {
    const { latitude, longitude } = station.astronomy;
    const marker = L.marker([latitude, longitude], { icon: blueIcon })
        .bindPopup(`<b>${station.station.name} (${station.station.code})</b>`)
        .addTo(layerGroups.stations);
    marker.stationId = id;
    marker.on('click', onClick);
    stationMarkers[id] = marker;
}

/**
 * Updates the icon for a specific station marker.
 * @param {string} id - The station ID.
 * @param {L.Icon} icon - The new Leaflet icon to apply.
 */
export function updateStationMarkerIcon(id, icon) {
    if (stationMarkers[id]) {
        stationMarkers[id].setIcon(icon);
    }
}

/**
 * Draws the ground track for a selected satellite or aircraft pass.
 * This is the function that now handles variable transparency for aircraft.
 * @param {string|null} id - The ID of the pass/crossing to highlight.
 * @param {string} type - 'satellite' or 'aircraft'.
 */
export function highlightTrack(id, type) {
    layerGroups.groundTracks.clearLayers();
    if (!id) return;

    const sourceData = type === 'satellite' ? passData.passes : aircraftData.crossings;
    const idKey = type === 'satellite' ? 'pass_id' : 'crossing_id';
    const item = sourceData?.find(p => p[idKey] === id);
    if (!item || !item.ground_track) return;

    const groundPoints = item.ground_track.map(p => [p.lat, p.lon]);

    if (type === 'aircraft' && item.station_sky_tracks) {
        // --- Aircraft Track with Variable Transparency ---
        // This logic draws the aircraft track as a series of small segments,
        // each colored with the opacity value calculated by the backend.
        const firstStationId = Object.keys(item.station_sky_tracks)[0];
        const skyTrack = item.station_sky_tracks[firstStationId];

        // Create a quick lookup map from timestamp to opacity
        const timeToOpacityMap = new Map(skyTrack.map(p => [p.time, p.opacity]));

        for (let i = 0; i < item.ground_track.length - 1; i++) {
            const point1 = item.ground_track[i];
            const point2 = item.ground_track[i + 1];
            const opacity = timeToOpacityMap.get(point1.time) ?? 1.0;

            L.polyline(
                [[point1.lat, point1.lon], [point2.lat, point2.lon]],
                { color: '#ff00ff', weight: 3, opacity: opacity }
            ).addTo(layerGroups.groundTracks);
        }
    } else {
        // --- Satellite Track with Uniform Transparency ---
        L.polyline(groundPoints, { color: '#ffff00', weight: 3, opacity: 0.8 }).addTo(layerGroups.groundTracks);
    }

    // Add start/end labels and fit map to bounds
    const startPoint = groundPoints[0];
    const endPoint = groundPoints[groundPoints.length - 1];
    const startTime = new Date(item.ground_track[0].time).toUTCString().substring(17, 22);
    const endTime = new Date(item.ground_track[item.ground_track.length - 1].time).toUTCString().substring(17, 22);

    L.marker(startPoint, { icon: L.divIcon({ className: 'ground-track-label', html: `S: ${startTime}` }) }).addTo(layerGroups.groundTracks);
    L.marker(endPoint, { icon: L.divIcon({ className: 'ground-track-label', html: `E: ${endTime}` }) }).addTo(layerGroups.groundTracks);

    map.flyToBounds(L.latLngBounds(groundPoints), { padding: [50, 50], maxZoom: 7 });
}

/**
 * Toggles the visibility of the meteor layer.
 * @param {boolean} show - Whether to show or hide the layer.
 * @param {function} onClick - Click handler for meteor tracks.
 * @param {function} onMouseover - Mouseover handler.
 * @param {function} onMouseout - Mouseout handler.
 */
export function toggleMeteorLayer(show, onClick, onMouseover, onMouseout) {
    if (show) {
        displayMeteors(onClick, onMouseover, onMouseout);
    } else {
        layerGroups.meteors.clearLayers();
    }
}

/**
 * Draws all meteor tracks on the map.
 */
export function displayMeteors(onClick, onMouseover, onMouseout) {
    layerGroups.meteors.clearLayers();
    meteorData.forEach(meteor => {
        const polyline = L.polyline([
            [meteor.lat1, meteor.lon1],
            [meteor.lat2, meteor.lon2]
        ], { color: 'orange', weight: 2 });

        polyline.on('click', () => onClick(meteor));
        polyline.on('mouseover', () => onMouseover(meteor));
        polyline.on('mouseout', () => onMouseout(meteor));
        polyline.addTo(layerGroups.meteors);
    });
}

/**
 * Draws all lightning strikes on the map.
 * @param {boolean} is24hFilter - Whether to filter for the last 24 hours.
 * @param {function} onStrikeClick - Click handler for strikes.
 */
export function displayLightningStrikes(is24hFilter, onStrikeClick) {
    layerGroups.lightning.clearLayers();
    let filtered = lightningData;
    if (is24hFilter) {
        const twentyFourHoursAgo = Date.now() - 24 * 60 * 60 * 1000;
        filtered = lightningData.filter(s => new Date(s.time).getTime() >= twentyFourHoursAgo);
    }

    filtered.forEach(strike => {
        const iconClass = strike.type === 'cg' ? 'lightning-icon-yellow' : 'lightning-icon-white';
        const icon = L.divIcon({
            className: `lightning-icon ${iconClass} icon-size-18`,
            html: '⚡'
        });
        const marker = L.marker([strike.lat, strike.lon], { icon: icon });
        marker.on('click', () => onStrikeClick(strike, true));
        marker.strikeData = strike;
        marker.addTo(layerGroups.lightning);
    });
}

/**
 * Highlights a specific lightning strike on the map.
 * @param {object} selectedStrike - The strike data object.
 * @param {boolean} shouldPan - Whether to pan the map to the strike.
 */
export function selectLightningStrikeOnMap(selectedStrike, shouldPan) {
    layerGroups.lightning.eachLayer(marker => {
        const isSelected = marker.strikeData.time === selectedStrike.time;
        const iconClass = marker.strikeData.type === 'cg' ? 'lightning-icon-yellow' : 'lightning-icon-white';
        const newClass = isSelected ? 'lightning-icon lightning-icon-red icon-size-24' : `lightning-icon ${iconClass} icon-size-18`;
        marker.getElement().className = newClass;
    });
    if (shouldPan) {
        map.flyTo([selectedStrike.lat, selectedStrike.lon], Math.max(map.getZoom(), 8));
    }
}

/**
 * Manages time-dependent overlays like terminator, clouds, and aurora.
 */
export function updateTimeDependentLayers(date, hour, minute) {
    if (layerGroups.timeLayers.terminator) {
        map.removeLayer(layerGroups.timeLayers.terminator);
        const time = new Date(`${date}T${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}:00Z`);
        layerGroups.timeLayers.terminator = L.terminator({ time: time }).addTo(map);
    }
    // Logic for other time-dependent layers can be added here
}

/**
 * Generic function to toggle a map layer.
 * @param {string} layerName - The name of the layer to toggle ('terminator', 'cloud', 'aurora').
 * @param {boolean} show - Whether to show or hide.
 */
export function toggleLayer(layerName, show, ...args) {
    const layer = layerGroups.timeLayers[layerName];
    if (layer) {
        map.removeLayer(layer);
        layerGroups.timeLayers[layerName] = null;
    }

    if (show) {
        if (layerName === 'terminator') {
            const [date, hour, minute] = args;
            const time = new Date(`${date}T${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}:00Z`);
            layerGroups.timeLayers.terminator = L.terminator({ time: time }).addTo(map);
        }
    }
}

// Functions for bearing lines (placeholders, as they depend on ui_manager)
export function drawBearingLines() { /* Logic would be added here */ }
export function clearBearingLines() { layerGroups.bearingLines.clearLayers(); }
