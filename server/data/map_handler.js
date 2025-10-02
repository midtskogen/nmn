import { createEl, TO_RAD } from './utils.js';
import { calculateBearing, destinationPoint, getSunAltitude } from './calculations.js';

// --- Module-scoped variables to hold map state ---
let map; // The main Leaflet map instance.
let cloudLayer = null, auroraLayer = null, terminatorLayer = null; // Holds time-dependent WMS/Grid layers.
let lightningLayer = null, selectedLightningMarker = null, meteorLayer = null; // Holds layers for specific data types.
let stationMarkers = {}, groundTrackLayers = {}, bearingLineLayer = null; // Holds station markers and track-related layers.
let passData = {}, aircraftData = {}, lightningData = [], meteorData = []; // Caches for data fetched from the backend.

// --- Icon Definitions ---
// Defines custom icons for map markers using Leaflet's L.Icon class.
const iconOptions = { iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41] };
export const blueIcon = new L.Icon({ iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png', ...iconOptions, shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png' });
export const redIcon = new L.Icon({ iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png', ...iconOptions, shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png' });
export const yellowIcon = new L.Icon({ iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-yellow.png', ...iconOptions, shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png' });
// Creates a custom HTML-based icon for lightning strikes.
const createLightningIcon = (color, size) => L.divIcon({ className: `lightning-icon lightning-icon-${color} icon-size-${size}`, html: '⚡', iconSize: [size, size], iconAnchor: [size / 2, size / 2] });

// --- Custom Leaflet Layer Extensions ---
/**
 * A custom TileLayer that inverts the colors of map label tiles, effectively
 * making a light-themed label layer suitable for a dark satellite map.
 */
L.TileLayer.WhiteToTransparent = L.TileLayer.extend({
    createTile: function (coords, done) {
        const tile = document.createElement('canvas');
        const size = this.getTileSize();
        tile.width = size.x*2;
        tile.height = size.y*2;
        const ctx = tile.getContext('2d');
        const img = document.createElement('img');

        img.onload = () => {
            ctx.drawImage(img, 0, 0);
            const imageData = ctx.getImageData(0, 0, size.x*2, size.y*2);
            const data = imageData.data;
            // Inverts the color of each pixel.
            for (let i = 0; i < data.length; i += 4) {
                data[i] = data[i + 1] = data[i + 2] = data[i + 3] = 255 - data[i];
            }
            ctx.putImageData(imageData, 0, 0);
            done(null, tile);
        };
        img.onerror = () => done(new Error('Tile load error'), tile);
        img.crossOrigin = "Anonymous";
        img.src = this.getTileUrl(coords);
        return tile;
    }
});
/**
 * A custom GridLayer that draws the day/night terminator and twilight zones on the map.
 * It uses an adaptive quadtree-based rendering algorithm to efficiently draw the smooth
 * twilight gradients by recursively subdividing the tile.
 */
L.GridLayer.Terminator = L.GridLayer.extend({
    options: { date: null },
    createTile: function (coords) {
        const tile = L.DomUtil.create('canvas', 'leaflet-tile');
        const size = this.getTileSize();
        tile.width = size.x;
        tile.height = size.y;
        const ctx = tile.getContext('2d');
        const date = this.options.date;
        if (!date) return tile;

        const nwPoint = coords.scaleBy(size);
        const map = this._map;
        const MAX_RESOLUTION = 64, MIN_RESOLUTION = 2; // Defines the recursive subdivision limits.

        // Helper to get lat/lon for a pixel within the tile.
        const getAltitude = (x, y) => map.unproject(nwPoint.add(L.point(x, y)), coords.z);
        // Categorizes a sun altitude into a twilight zone.
        const getAltitudeZone = (alt) => {
            if (alt >= 0) return 0;      // Day
            if (alt >= -6) return 1;     // Civil Twilight
            if (alt >= -12) return 2;    // Nautical Twilight
            if (alt >= -18) return 3;    // Astronomical Twilight
            return 4;                   // Full Night
        };
        // Returns the appropriate semi-transparent color for a given twilight zone.
        const getFillStyleForZone = (zone) => {
            switch(zone) {
                case 1: return 'rgba(0, 0, 50, 0.15)';
                case 2: return 'rgba(0, 0, 50, 0.30)';
                case 3: return 'rgba(0, 0, 50, 0.45)';
                case 4: return 'rgba(0, 0, 50, 0.60)';
                default: return null;
            }
        };
        /**
         * The core recursive rendering function. If all four corners of a block are in the same
         * twilight zone, it fills the block with a solid color. Otherwise, it subdivides the
         * block into four smaller blocks and calls itself for each.
         */
        const renderBlock = (x, y, size) => {
            const p = getAltitude(x, y);
            const zone_tl = getAltitudeZone(getSunAltitude(date, p.lat, p.lng));
            if (size <= MIN_RESOLUTION) {
                const fill = getFillStyleForZone(zone_tl);
                if (fill) { ctx.fillStyle = fill; ctx.fillRect(x, y, size, size); }
                return;
            }
            const p_tr = getAltitude(x + size, y), p_bl = getAltitude(x, y + size), p_br = getAltitude(x + size, y + size);
            const zone_tr = getAltitudeZone(getSunAltitude(date, p_tr.lat, p_tr.lng));
            const zone_bl = getAltitudeZone(getSunAltitude(date, p_bl.lat, p_bl.lng));
            const zone_br = getAltitudeZone(getSunAltitude(date, p_br.lat, p_br.lng));

            if (zone_tl === zone_tr && zone_tl === zone_bl && zone_tl === zone_br) {
                const fill = getFillStyleForZone(zone_tl);
                if (fill) { ctx.fillStyle = fill; ctx.fillRect(x, y, size, size); }
            } else {
                const newSize = size / 2;
                renderBlock(x, y, newSize); renderBlock(x + newSize, y, newSize);
                renderBlock(x, y + newSize, newSize); renderBlock(x + newSize, y + newSize, newSize);
            }
        };
        // Kicks off the rendering process for the entire tile.
        for (let y = 0; y < size.y; y += MAX_RESOLUTION) {
            for (let x = 0; x < size.x; x += MAX_RESOLUTION) { renderBlock(x, y, MAX_RESOLUTION); }
        }
        return tile;
    },
    // Method to update the date for the terminator calculation and trigger a redraw.
    setDate: function(date) { this.options.date = date; this.redraw(); }
});
// --- Public API ---

/**
 * Initializes the Leaflet map, its base layers, custom controls, and event listeners.
 * @param {string} mapId - The ID of the div element where the map will be rendered.
 * @param {function} onMoveEnd - Callback function to execute when the map finishes panning.
 * @param {function} onZoomEnd - Callback function to execute when the map finishes zooming.
 * @returns {L.Map} The initialized Leaflet map instance.
 */
export function initMap(mapId, onMoveEnd, onZoomEnd) {
    const defaultMapView = [[64.7, 13.0], 5];
    map = L.map(mapId, { maxZoom: 12, minZoom: 3 }).setView(...defaultMapView);
    
    // Add base satellite and inverted label layers.
    L.tileLayer(`https://api.maptiler.com/maps/satellite/{z}/{x}/{y}.jpg?key=2gLVEsGCx9JWRMUG7191`, {
        attribution: '<a href="https://www.maptiler.com/copyright/" target="_blank">&copy; MapTiler</a> <a href="https://www.openstreetmap.org/copyright" target="_blank">&copy; OpenStreetMap contributors</a>',
        className: 'map-tiler-satellite'
    }).addTo(map);
    map.createPane('labels').style.zIndex = 650;
    map.getPane('labels').style.pointerEvents = 'none';
    map.createPane('meteorTooltipPane').style.zIndex = 651;

    new L.TileLayer.WhiteToTransparent(`https://api.maptiler.com/maps/backdrop/{z}/{x}/{y}@2x.png?key=2gLVEsGCx9JWRMUG7191`, {
        attribution: '', tileSize: 512, zoomOffset: -1
    }).addTo(map);

    // Add a custom control to reset the map view to its default position.
    L.Control.ResetView = L.Control.extend({
        options: { position: 'topright' },
        onAdd: (map) => createEl('button', { className: 'leaflet-reset-view-control', innerHTML: 'Tilbakestill', onclick: (e) => { e.stopPropagation(); map.setView(...defaultMapView); } })
    });
    new L.Control.ResetView().addTo(map);

    // Bind map events to the provided callback functions.
    map.on('moveend', onMoveEnd);
    map.on('zoomend', onZoomEnd);
    return map;
}

// --- Data Setters ---
// These functions allow other modules to provide data to this handler.
export function setPassData(data) { passData = data; }
export function setAircraftData(data) { aircraftData = data; }
export function setLightningData(data) { lightningData = data; }
export function setMeteorData(data) { meteorData = data; }


// --- Marker Management ---
/** Adds a station marker to the map. */
export function addStationMarker(stationId, station, onClick) {
    const marker = L.marker([station.astronomy.latitude, station.astronomy.longitude], { icon: blueIcon }).addTo(map);
    marker.stationId = stationId;
    marker.bindTooltip(`<b>${station.station.code}</b><br>${station.station.name}`).on('click', onClick);
    stationMarkers[stationId] = marker;
    return marker;
}
/** Updates the icon of a specific station marker. */
export function updateStationMarkerIcon(stationId, icon) { if (stationMarkers[stationId]) stationMarkers[stationId].setIcon(icon); }
/** Returns the object containing all station marker instances. */
export function getStationMarkers() { return stationMarkers; }


// --- Layer & Drawing Functions ---

/**
 * Draws timestamp labels along a satellite or aircraft ground track. It implements
 * a simple decluttering algorithm to avoid drawing labels too close to each other
 * at high zoom levels.
 * @param {object} item - The pass or crossing object containing the `ground_track` array.
 * @param {string} [color='#FF0000'] - The color for the time markers.
 * @returns {L.LayerGroup} A layer group containing the new time markers.
 */
const drawTimeMarkers = (item, color = '#FF0000') => {
    const layer = L.layerGroup();
    if (!item.ground_track || item.ground_track.length === 0) return layer;

    const minPixelDistance = 100; // Minimum screen distance in pixels between labels.
    let lastLabelPoint = null;

    item.ground_track.forEach(point => {
        const latLng = L.latLng(point.lat, point.lon);
        const currentPoint = map.latLngToContainerPoint(latLng);

        let shouldDrawLabel = false;
        if (lastLabelPoint === null) {
            shouldDrawLabel = true; // Always draw the first label.
        } else {
            const distance = currentPoint.distanceTo(lastLabelPoint);
            if (distance > minPixelDistance) {
                shouldDrawLabel = true;
            }
        }

        if (shouldDrawLabel) {
            const date = new Date(point.time);
            const timeString = `${String(date.getUTCHours()).padStart(2, '0')}:${String(date.getUTCMinutes()).padStart(2, '0')}:${String(date.getUTCSeconds()).padStart(2, '0')}`;
            L.circleMarker([point.lat, point.lon], { radius: 3, color: color, fillColor: color, fillOpacity: 1 }).addTo(layer);
            L.marker([point.lat, point.lon], {
                icon: L.divIcon({ className: 'ground-track-label', html: `<span>${timeString}</span>`, iconSize: [80, 12], iconAnchor: [-5, 6] })
            }).addTo(layer);
            lastLabelPoint = currentPoint;
        }
    });
    return layer;
};
/**
 * Creates a polyline for a ground track. If there are large time gaps in the
 * track data (more than 15 minutes), it breaks the line into multiple segments.
 * @param {Array<object>} groundTrack - An array of track points.
 * @param {object} style - Leaflet path styling options.
 * @returns {L.FeatureGroup} A feature group containing the polyline segments.
 */
const createSegmentedPolyline = (groundTrack, style) => {
    const featureGroup = L.featureGroup();
    if (!groundTrack || groundTrack.length < 2) {
        return featureGroup; // Return empty group if not enough points for a line.
    }

    let currentSegment = [[groundTrack[0].lat, groundTrack[0].lon]];
    const fifteenMinutesInMs = 15 * 60 * 1000;

    for (let i = 1; i < groundTrack.length; i++) {
        const prevPoint = groundTrack[i - 1];
        const currentPoint = groundTrack[i];
        const prevTime = new Date(prevPoint.time).getTime();
        const currentTime = new Date(currentPoint.time).getTime();

        if (currentTime - prevTime > fifteenMinutesInMs) {
            // Gap detected, finalize the previous segment.
            if (currentSegment.length > 1) {
                L.polyline(currentSegment, style).addTo(featureGroup);
            }
            // Start a new segment.
            currentSegment = [];
        }
        currentSegment.push([currentPoint.lat, currentPoint.lon]);
    }

    // Add the last segment.
    if (currentSegment.length > 1) {
        L.polyline(currentSegment, style).addTo(featureGroup);
    }

    return featureGroup;
};


/**
 * Draws lines from a selected station to the start and end points of a pass/crossing
 * as seen by the selected cameras.
 * @param {object} item - The pass or crossing data object.
 * @param {Array<object>} events - The selected camera view events.
 * @param {object} stationsData - The main station data object.
 */
export function drawBearingLines(item, events, stationsData) {
    if (bearingLineLayer) map.removeLayer(bearingLineLayer);
    bearingLineLayer = L.layerGroup().addTo(map);
    events.forEach(event => {
        const station = stationsData[event.station_id];
        if (!station) return;
        const stationCoords = [station.astronomy.latitude, station.astronomy.longitude];
        // Find the points on the ground track that are closest in time to the event's start and end.
        let startPoint = item.ground_track[0], endPoint = item.ground_track[item.ground_track.length - 1];
        let minStartDiff = Infinity, minEndDiff = Infinity;
        const startTime = new Date(event.start_utc).getTime(), endTime = new Date(event.end_utc).getTime();
        item.ground_track.forEach(p => {
            const pointTime = new Date(p.time).getTime();
            const startDiff = Math.abs(pointTime - startTime), endDiff = Math.abs(pointTime - endTime);
            if(startDiff < minStartDiff) { minStartDiff = startDiff; startPoint = p; }
            if(endDiff < minEndDiff) { minEndDiff = endDiff; endPoint = p; }
        });
        // Draw dashed lines to these points.
        L.polyline([stationCoords, [startPoint.lat, startPoint.lon]], { color: '#FFFF00', weight: 2, dashArray: '5, 5' }).addTo(bearingLineLayer);
        L.polyline([stationCoords, [endPoint.lat, endPoint.lon]], { color: '#FFFF00', weight: 2, dashArray: '5, 5' }).addTo(bearingLineLayer);
    });
}
/** Removes all bearing lines from the map. */
export function clearBearingLines() {
    if (bearingLineLayer) {
        map.removeLayer(bearingLineLayer);
        bearingLineLayer = null;
    }
}

/**
 * Manages the drawing and highlighting of all satellite and aircraft tracks on the map.
 * @param {string|null} id - The ID of the track to highlight. If null, all tracks are drawn un-highlighted.
 * @param {string} type - The type of track to highlight ('satellite' or 'aircraft').
 * @param {boolean} isSatView - Whether the satellite view is currently active.
 * @param {boolean} isAircraftView - Whether the aircraft view is currently active.
 */
export function highlightTrack(id, type, isSatView, isAircraftView) {
    // Clear all existing track layers.
    Object.values(groundTrackLayers).forEach(layer => { if (layer) map.removeLayer(layer); });
    if (bearingLineLayer) map.removeLayer(bearingLineLayer);
    groundTrackLayers = {};
    bearingLineLayer = null;
    
    // Draw all satellite passes.
    if (passData.passes) {
        passData.passes.forEach(pass => {
            const isHighlighted = isSatView && pass.pass_id === id;
            const style = { color: isHighlighted ? 'red' : 'grey', weight: isHighlighted ? 3 : 2, opacity: isHighlighted ? 1.0 : 0.6 };
            const track = createSegmentedPolyline(pass.ground_track, style);
            const timeMarkers = drawTimeMarkers(pass, 'red');
            groundTrackLayers[pass.pass_id] = track;
            groundTrackLayers[pass.pass_id + '_markers'] = timeMarkers;
            if (isSatView) {
                track.addTo(map);
                if (isHighlighted) timeMarkers.addTo(map); // Only show time markers for the highlighted track.
            }
        });
    }

    // Draw all aircraft crossings.
    if (aircraftData.crossings) {
        aircraftData.crossings.forEach(crossing => {
            const isHighlighted = isAircraftView && crossing.crossing_id === id;
            const style = { color: isHighlighted ? 'yellow' : '#a0522d', weight: isHighlighted ? 3 : 2, opacity: isHighlighted ? 1.0 : 0.6 };
            const track = createSegmentedPolyline(crossing.ground_track, style);
            const timeMarkers = drawTimeMarkers(crossing, 'yellow');
            groundTrackLayers[crossing.crossing_id] = track;
            groundTrackLayers[crossing.crossing_id + '_markers'] = timeMarkers;
            if (isAircraftView) {
                track.addTo(map);
                if (isHighlighted) {
                    timeMarkers.addTo(map);
                }
            }
        });
    }
}

/**
 * Draws lightning strike markers on the map.
 * @param {object} stationInfo - Not directly used, but kept for potential future use.
 * @param {boolean} is24hFilter - Whether to filter strikes to the last 24 hours.
 * @param {function} onStrikeSelect - Callback for when a strike marker is clicked.
 */
export function displayLightningStrikes(stationInfo, is24hFilter, onStrikeSelect) {
    if (lightningLayer) map.removeLayer(lightningLayer);
    lightningLayer = L.layerGroup();
    // Filter the data based on the 24-hour toggle.
    const strikes = lightningData.filter(strike => {
        if (!is24hFilter) return true;
        const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
        return new Date(strike.time) >= twentyFourHoursAgo;
    });
    strikes.forEach((strike, index) => {
        const { type, dist, lat, lon } = strike;
        const iconColor = type === 'ic' ? 'white' : 'yellow', iconSize = type === 'ic' ? 14 : 18;
        const opacity = Math.max(0.1, 1.0 - (dist / 30.0));
        const marker = L.marker([lat, lon], { icon: createLightningIcon(iconColor, iconSize), opacity: opacity });
        strike.id = `lightning-${index}`; // Ensure ID is set for linking with the UI panel.
        marker.strike = strike;
        marker.bindTooltip(`${new Date(strike.time).toISOString().slice(0,19).replace('T',' ')} UTC`);
        marker.on('click', () => onStrikeSelect(strike, marker, false));
        lightningLayer.addLayer(marker);
    });
    // Add the layer to the map only if the main lightning toggle is active.
    if (document.getElementById('lightning-toggle').checked) {
        lightningLayer.addTo(map);
    }
}

/**
 * Visually highlights a specific lightning strike marker on the map.
 * @param {L.Marker} marker - The marker instance to highlight.
 * @param {boolean} shouldPan - Whether to pan the map to the marker's location.
 */
export function selectLightningStrikeOnMap(marker, shouldPan) {
    // Reset the previously selected marker, if any.
    if (selectedLightningMarker) {
        const { type } = selectedLightningMarker.strike;
        selectedLightningMarker.setIcon(createLightningIcon(type === 'ic' ? 'white' : 'yellow', type === 'ic' ? 14 : 18));
    }
    // Set the new marker's icon to the highlighted state.
    marker.setIcon(createLightningIcon('red', 24));
    selectedLightningMarker = marker;
    if (shouldPan) {
        map.setView([marker.strike.lat, marker.strike.lon], 8);
    }
}

/**
 * Draws meteor tracks on the map. Tracks are rendered as polygons whose width
 * is proportional to the meteor's altitude and the current map zoom level,
 * creating a pseudo-3D effect.
 * @param {function} onMeteorClick - Callback for click events.
 * @param {function} onMeteorMouseover - Callback for mouseover events.
 * @param {function} onMeteorMouseout - Callback for mouseout events.
 */
export function displayMeteors(onMeteorClick, onMeteorMouseover, onMeteorMouseout) {
    if (meteorLayer) map.removeLayer(meteorLayer);
    meteorLayer = L.layerGroup();
    const zoom = map.getZoom();
    // Calculate the approximate number of meters per pixel at the map's center latitude and current zoom.
    const metersPerPixel = 40075016.686 * Math.abs(Math.cos(map.getCenter().lat * TO_RAD)) / Math.pow(2, zoom + 8);
    // Function to determine the desired width of the meteor polygon based on altitude.
    const getWidthInMeters = (h, isEnd = false) => (1 + (1 - (Math.min(100, Math.max(0, h)) / 100)) * 4 + (isEnd ? 3 : 0)) * metersPerPixel;

    meteorData.forEach(meteor => {
        const width1_m = getWidthInMeters(meteor.h1), width2_m = getWidthInMeters(meteor.h2, true);
        const bearing = calculateBearing(meteor.lat1, meteor.lon1, meteor.lat2, meteor.lon2);
        // Calculate the four corner points of the trapezoidal polygon.
        const perpBearing1 = (bearing + 90) % 360, perpBearing2 = (bearing - 90 + 360) % 360;
        const p1_left = destinationPoint(meteor.lat1, meteor.lon1, width1_m / 2, perpBearing2);
        const p1_right = destinationPoint(meteor.lat1, meteor.lon1, width1_m / 2, perpBearing1);
        const p2_left = destinationPoint(meteor.lat2, meteor.lon2, width2_m / 2, perpBearing2);
        const p2_right = destinationPoint(meteor.lat2, meteor.lon2, width2_m / 2, perpBearing1);
        const meteorPolygon = L.polygon([p1_left, p1_right, p2_right, p2_left], { color: '#ff9900', fillColor: '#ff9900', weight: 0, fillOpacity: 0.7 });
        // Add a circular end cap to the polygon for a rounded look.
        const endCap = L.circle([meteor.lat2, meteor.lon2], { radius: width2_m / 2, color: '#ff9900', fillColor: '#ff9900', weight: 0, fillOpacity: 0.7 });
        const meteorShape = L.featureGroup([meteorPolygon, endCap]);
        meteorShape.bindTooltip(`Meteor: ${meteor.timestamp.replace('T', ' ').replace('Z', ' UTC')}`, { pane: 'meteorTooltipPane' });
        meteorShape.on('click', () => onMeteorClick(meteor));
        meteorShape.on('mouseover', () => onMeteorMouseover(meteor));
        meteorShape.on('mouseout', () => onMeteorMouseout(meteor));
        meteorShape.addTo(meteorLayer);
    });
    meteorLayer.addTo(map);
}

/**
 * Handles the showing and hiding of the meteor data layer.
 * @param {boolean} isChecked - The state of the toggle switch.
 * @param {function} onMeteorClick - Callback for click events.
 * @param {function} onMeteorMouseover - Callback for mouseover events.
 * @param {function} onMeteorMouseout - Callback for mouseout events.
 */
export function toggleMeteorLayer(isChecked, onMeteorClick, onMeteorMouseover, onMeteorMouseout) {
    if (isChecked) {
        // If turning the layer on, call the main display function which handles drawing.
        displayMeteors(onMeteorClick, onMeteorMouseover, onMeteorMouseout);
    } else {
        // If turning the layer off, simply remove the layer from the map if it exists.
        if (meteorLayer && map.hasLayer(meteorLayer)) {
            map.removeLayer(meteorLayer);
        }
    }
}

/** Returns the main Leaflet map instance. */
export function getMap() { return map; }

/** Lazily creates and returns the singleton terminator layer instance. */
function getTerminatorLayer() { 
    if (!terminatorLayer) terminatorLayer = new L.GridLayer.Terminator();
    return terminatorLayer;
}
/** Creates or updates and returns the cloud WMS layer for a specific date. */
function getCloudLayer(date) { 
    if (cloudLayer && cloudLayer.wmsParams.time === date) return cloudLayer;
    if (cloudLayer) map.removeLayer(cloudLayer);
    cloudLayer = L.tileLayer.wms('https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi', { layers: 'VIIRS_SNPP_CorrectedReflectance_TrueColor', format: 'image/png', transparent: true, time: date, attribution: `NASA GIBS | ${date}` });
    cloudLayer.setOpacity(0.5);
    return cloudLayer;
}
/** Creates or updates and returns the aurora (VIIRS DNB) WMS layer for a specific date. */
function getAuroraLayer(date) { 
    if (auroraLayer && auroraLayer.wmsParams.time === date) return auroraLayer;
    if (auroraLayer) map.removeLayer(auroraLayer);
    auroraLayer = L.tileLayer.wms('https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi', { layers: 'VIIRS_SNPP_DayNightBand_At_Sensor_Radiance', format: 'image/png', transparent: true, time: date, attribution: `NASA GIBS | ${date}` });
    auroraLayer.setOpacity(0.5);
    return auroraLayer;
}

/**
 * A generic function to toggle the visibility of various map overlay layers.
 * @param {string} layerType - The type of layer to toggle ('cloud', 'aurora', 'terminator', 'lightning').
 * @param {boolean} isChecked - The desired visibility state.
 * @param {string} [date] - The date string, required for time-dependent layers.
 * @param {string} [hour] - The hour, required for the terminator layer.
 * @param {string} [minute] - The minute, required for the terminator layer.
 */
export function toggleLayer(layerType, isChecked, date, hour, minute) {
    // This block handles the simple visibility toggling for the pre-rendered lightning layer.
    if (layerType === 'lightning') {
        if (isChecked) {
            if (lightningLayer && !map.hasLayer(lightningLayer)) {
                map.addLayer(lightningLayer);
            }
        } else {
            if (lightningLayer && map.hasLayer(lightningLayer)) {
                map.removeLayer(lightningLayer);
            }
        }
        return;
    }
    
    // This block handles the more complex WMS/Grid layers that need to be created or updated.
    const layerGetters = {
        cloud: () => getCloudLayer(date),
        aurora: () => getAuroraLayer(date),
        terminator: () => getTerminatorLayer()
    };
    const activeLayers = {
        cloud: cloudLayer,
        aurora: auroraLayer,
        terminator: terminatorLayer
    };
    let layer = activeLayers[layerType];

    if (isChecked) {
        // Ensure the layer object exists before trying to add it.
        if (!layerGetters[layerType]) return;
        layer = layerGetters[layerType]();
        
        // The terminator layer needs the specific time to be set.
        if (layerType === 'terminator') {
            const currentTime = new Date(`${date}T${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}:00Z`);
            layer.setDate(currentTime);
        }

        if (!map.hasLayer(layer)) {
            map.addLayer(layer);
        }
    } else {
        if (layer && map.hasLayer(layer)) {
            map.removeLayer(layer);
        }
    }
}

/**
 * A utility function called when the time in the form changes. It ensures that
 * all active time-dependent layers are updated to reflect the new time.
 * @param {string} date
 * @param {string} hour
 * @param {string} minute
 */
export function updateTimeDependentLayers(date, hour, minute) {
    if (document.getElementById('cloud-toggle').checked) {
        toggleLayer('cloud', true, date);
    }
    if (document.getElementById('aurora-toggle').checked) {
        toggleLayer('aurora', true, date);
    }
    if (document.getElementById('terminator-toggle').checked) {
        toggleLayer('terminator', true, date, hour, minute);
    }
}
