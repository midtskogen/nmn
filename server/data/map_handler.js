import { createEl, TO_RAD } from './utils.js';
import { calculateBearing, destinationPoint } from './calculations.js';

// --- Module-scoped variables to hold map state ---
let map;
let cloudLayer = null, auroraLayer = null, terminatorLayer = null;
let lightningLayer = null, selectedLightningMarker = null, meteorLayer = null;
let stationMarkers = {}, groundTrackLayers = {}, bearingLineLayer = null;
let passData = {}, aircraftData = {}, lightningData = [], meteorData = [];

// --- Icon Definitions ---
const iconOptions = { iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41] };
export const blueIcon = new L.Icon({ iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png', ...iconOptions, shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png' });
export const redIcon = new L.Icon({ iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png', ...iconOptions, shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png' });
export const yellowIcon = new L.Icon({ iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-yellow.png', ...iconOptions, shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png' });
const createLightningIcon = (color, size) => L.divIcon({ className: `lightning-icon lightning-icon-${color} icon-size-${size}`, html: '⚡', iconSize: [size, size], iconAnchor: [size / 2, size / 2] });

// --- Custom Leaflet Layer Extensions ---
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
        const MAX_RESOLUTION = 64, MIN_RESOLUTION = 2;

        const getAltitude = (x, y) => map.unproject(nwPoint.add(L.point(x, y)), coords.z);
        const getAltitudeZone = (alt) => {
            if (alt >= 0) return 0;      // Day
            if (alt >= -6) return 1;     // Civil Twilight
            if (alt >= -12) return 2;    // Nautical Twilight
            if (alt >= -18) return 3;    // Astronomical Twilight
            return 4;                    // Full Night
        };
        
        const getFillStyleForZone = (zone) => {
            switch(zone) {
                case 1: return 'rgba(0, 0, 50, 0.15)';
                case 2: return 'rgba(0, 0, 50, 0.30)';
                case 3: return 'rgba(0, 0, 50, 0.45)';
                case 4: return 'rgba(0, 0, 50, 0.60)';
                default: return null;
            }
        };
        
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
        
        for (let y = 0; y < size.y; y += MAX_RESOLUTION) {
            for (let x = 0; x < size.x; x += MAX_RESOLUTION) { renderBlock(x, y, MAX_RESOLUTION); }
        }
        return tile;
    },
    setDate: function(date) { this.options.date = date; this.redraw(); }
});

// --- Public API ---

export function initMap(mapId, onMoveEnd, onZoomEnd, t) {
    const defaultMapView = [[64.7, 13.0], 5];
    map = L.map(mapId, { maxZoom: 12, minZoom: 3 }).setView(...defaultMapView);
    
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
    
    L.Control.ResetView = L.Control.extend({
        options: { position: 'topright' },
        onAdd: (map) => createEl('button', { className: 'leaflet-reset-view-control', innerHTML: t('reset_button'), onclick: (e) => { e.stopPropagation(); map.setView(...defaultMapView); } })
    });
    new L.Control.ResetView().addTo(map);

    map.on('moveend', onMoveEnd);
    map.on('zoomend', onZoomEnd);
    return map;
}

// --- Data Setters ---
export function setPassData(data) { passData = data; }
export function setAircraftData(data) { aircraftData = data; }
export function setLightningData(data) { lightningData = data; }
export function setMeteorData(data) { meteorData = data; }

// --- Marker Management ---
export function addStationMarker(stationId, station, onClick) {
    const marker = L.marker([station.astronomy.latitude, station.astronomy.longitude], { icon: blueIcon }).addTo(map);
    marker.stationId = stationId;
    marker.bindTooltip(`<b>${station.station.code}</b><br>${station.station.name}`).on('click', onClick);
    stationMarkers[stationId] = marker;
    return marker;
}
export function updateStationMarkerIcon(stationId, icon) { if (stationMarkers[stationId]) stationMarkers[stationId].setIcon(icon); }
export function getStationMarkers() { return stationMarkers; }

// --- Layer & Drawing Functions ---

const drawTimeMarkers = (item, color = '#FF0000') => {
    const layer = L.layerGroup();
    if (!item.ground_track || item.ground_track.length === 0) return layer;

    const minPixelDistance = 100;
    let lastLabelPoint = null;

    item.ground_track.forEach(point => {
        const latLng = L.latLng(point.lat, point.lon);
        const currentPoint = map.latLngToContainerPoint(latLng);

        let shouldDrawLabel = false;
        if (lastLabelPoint === null) {
            shouldDrawLabel = true;
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

const createSegmentedPolyline = (groundTrack, style) => {
    const featureGroup = L.featureGroup();
    if (!groundTrack || groundTrack.length < 2) {
        return featureGroup;
    }

    let currentSegment = [[groundTrack[0].lat, groundTrack[0].lon]];
    const fifteenMinutesInMs = 15 * 60 * 1000;

    for (let i = 1; i < groundTrack.length; i++) {
        const prevPoint = groundTrack[i - 1];
        const currentPoint = groundTrack[i];
        const prevTime = new Date(prevPoint.time).getTime();
        const currentTime = new Date(currentPoint.time).getTime();
        if (currentTime - prevTime > fifteenMinutesInMs) {
            if (currentSegment.length > 1) {
                L.polyline(currentSegment, style).addTo(featureGroup);
            }
            currentSegment = [];
        }
        currentSegment.push([currentPoint.lat, currentPoint.lon]);
    }

    if (currentSegment.length > 1) {
        L.polyline(currentSegment, style).addTo(featureGroup);
    }

    return featureGroup;
};

export function drawBearingLines(item, events, stationsData) {
    if (bearingLineLayer) map.removeLayer(bearingLineLayer);
    bearingLineLayer = L.layerGroup().addTo(map);
    events.forEach(event => {
        const station = stationsData[event.station_id];
        if (!station) return;
        const stationCoords = [station.astronomy.latitude, station.astronomy.longitude];
        
        let startPoint = item.ground_track[0], endPoint = item.ground_track[item.ground_track.length - 1];
        let minStartDiff = Infinity, minEndDiff = Infinity;
        const startTime = new Date(event.start_utc).getTime(), endTime = new Date(event.end_utc).getTime();
        item.ground_track.forEach(p => {
            const pointTime = new Date(p.time).getTime();
            const startDiff = Math.abs(pointTime - startTime), endDiff = Math.abs(pointTime - endTime);
            if(startDiff < minStartDiff) { minStartDiff = startDiff; startPoint = p; }
            if(endDiff < minEndDiff) { minEndDiff = endDiff; endPoint = p; }
        });
        
        L.polyline([stationCoords, [startPoint.lat, startPoint.lon]], { color: '#FFFF00', weight: 2, dashArray: '5, 5' }).addTo(bearingLineLayer);
        L.polyline([stationCoords, [endPoint.lat, endPoint.lon]], { color: '#FFFF00', weight: 2, dashArray: '5, 5' }).addTo(bearingLineLayer);
    });
}

export function clearBearingLines() {
    if (bearingLineLayer) {
        map.removeLayer(bearingLineLayer);
        bearingLineLayer = null;
    }
}

export function highlightTrack(id, type, isSatView, isAircraftView) {
    Object.values(groundTrackLayers).forEach(layer => { if (layer) map.removeLayer(layer); });
    if (bearingLineLayer) map.removeLayer(bearingLineLayer);
    groundTrackLayers = {};
    bearingLineLayer = null;
    
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
                if (isHighlighted) timeMarkers.addTo(map);
            }
        });
    }

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

export function displayLightningStrikes(is24hFilter, onStrikeSelect) {
    if (lightningLayer) map.removeLayer(lightningLayer);
    lightningLayer = L.layerGroup();
    
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
        strike.id = `lightning-${index}`;
        marker.strike = strike;
        marker.bindTooltip(`${new Date(strike.time).toISOString().slice(0,19).replace('T',' ')} UTC`);
        marker.on('click', () => onStrikeSelect(strike, false));
        lightningLayer.addLayer(marker);
    });
    
    if (document.getElementById('lightning-toggle').checked) {
        lightningLayer.addTo(map);
    }
}

export function selectLightningStrikeOnMap(strike, shouldPan) {
    if (selectedLightningMarker) {
        const { type } = selectedLightningMarker.strike;
        selectedLightningMarker.setIcon(createLightningIcon(type === 'ic' ? 'white' : 'yellow', type === 'ic' ? 14 : 18));
    }
    
    let markerToSelect;
    if (lightningLayer) {
        lightningLayer.eachLayer(marker => {
            if (marker.strike.id === strike.id) {
                markerToSelect = marker;
            }
        });
    }

    if (markerToSelect) {
        markerToSelect.setIcon(createLightningIcon('red', 24));
        selectedLightningMarker = markerToSelect;
        if (shouldPan) {
            map.setView([markerToSelect.strike.lat, markerToSelect.strike.lon], 8);
        }
    }
}

export function displayMeteors(onMeteorClick, onMeteorMouseover, onMeteorMouseout) {
    if (meteorLayer) map.removeLayer(meteorLayer);
    meteorLayer = L.layerGroup();
    const zoom = map.getZoom();
    const metersPerPixel = 40075016.686 * Math.abs(Math.cos(map.getCenter().lat * TO_RAD)) / Math.pow(2, zoom + 8);
    
    const getWidthInMeters = (h, isEnd = false) => (1 + (1 - (Math.min(100, Math.max(0, h)) / 100)) * 4 + (isEnd ? 3 : 0)) * metersPerPixel;
    
    meteorData.forEach(meteor => {
        const width1_m = getWidthInMeters(meteor.h1), width2_m = getWidthInMeters(meteor.h2, true);
        const bearing = calculateBearing(meteor.lat1, meteor.lon1, meteor.lat2, meteor.lon2);
        
        const perpBearing1 = (bearing + 90) % 360, perpBearing2 = (bearing - 90 + 360) % 360;
        const p1_left = destinationPoint(meteor.lat1, meteor.lon1, width1_m / 2, perpBearing2);
        const p1_right = destinationPoint(meteor.lat1, meteor.lon1, width1_m / 2, perpBearing1);
        const p2_left = destinationPoint(meteor.lat2, meteor.lon2, width2_m / 2, perpBearing2);
        const p2_right = destinationPoint(meteor.lat2, meteor.lon2, width2_m / 2, perpBearing1);
        const meteorPolygon = L.polygon([p1_left, p1_right, p2_right, p2_left], { color: '#ff9900', fillColor: '#ff9900', weight: 0, fillOpacity: 0.7 });
        
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

export function toggleMeteorLayer(isChecked, onMeteorClick, onMeteorMouseover, onMeteorMouseout) {
    if (isChecked) {
        displayMeteors(onMeteorClick, onMeteorMouseover, onMeteorMouseout);
    } else {
        if (meteorLayer && map.hasLayer(meteorLayer)) {
            map.removeLayer(meteorLayer);
        }
    }
}

export function getMap() { return map; }

function getTerminatorLayer() { 
    if (!terminatorLayer) terminatorLayer = new L.GridLayer.Terminator();
    return terminatorLayer;
}

function getCloudLayer(date) { 
    if (cloudLayer && cloudLayer.wmsParams.time === date) return cloudLayer;
    if (cloudLayer) map.removeLayer(cloudLayer);
    cloudLayer = L.tileLayer.wms('https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi', { layers: 'VIIRS_SNPP_CorrectedReflectance_TrueColor', format: 'image/png', transparent: true, time: date, attribution: `NASA GIBS | ${date}` });
    cloudLayer.setOpacity(0.5);
    return cloudLayer;
}

function getAuroraLayer(date) { 
    if (auroraLayer && auroraLayer.wmsParams.time === date) return auroraLayer;
    if (auroraLayer) map.removeLayer(auroraLayer);
    auroraLayer = L.tileLayer.wms('https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi', { layers: 'VIIRS_SNPP_DayNightBand_At_Sensor_Radiance', format: 'image/png', transparent: true, time: date, attribution: `NASA GIBS | ${date}` });
    auroraLayer.setOpacity(0.5);
    return auroraLayer;
}

export function toggleLayer(layerType, isChecked, date, hour, minute) {
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
        if (!layerGetters[layerType]) return;
        layer = layerGetters[layerType]();
        
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
