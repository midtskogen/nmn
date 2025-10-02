import { TO_RAD, TO_DEG } from './utils.js';

/**
 * Calculates the sunrise and sunset times for a given UTC date and geographical location.
 * This function can also calculate twilight times by adjusting the sun's altitude.
 * @param {Date} date The date for which to perform the calculation.
 * @param {number} lat Latitude in degrees.
 * @param {number} lon Longitude in degrees.
 * @param {number} [altitude=-6] The desired sun's altitude in degrees to define "rise" and "set".
 * Defaults to -6 degrees for civil twilight. Use 0 for official sunrise/sunset, -12 for nautical, and -18 for astronomical.
 * @returns {object} An object with `rise` (Date object or null), `set` (Date object or null), and `type` ('normal', 'polar_day', 'polar_night').
 */
export function getSunTimes(date, lat, lon, altitude = -6) {
    // Calculate the day of the year (1-366).
    const dayOfYear = (Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()) - Date.UTC(date.getUTCFullYear(), 0, 0)) / 864e5;
    
    // An approximation of the solar declination angle in radians.
    const solarDeclination = TO_DEG * (0.006918 - 0.399912 * Math.cos(2 * Math.PI * (dayOfYear - 1) / 365.24)
        + 0.070257 * Math.sin(2 * Math.PI * (dayOfYear - 1) / 365.24)
        - 0.006758 * Math.cos(4 * Math.PI * (dayOfYear - 1) / 365.24)
        + 0.000907 * Math.sin(4 * Math.PI * (dayOfYear - 1) / 365.24)
        - 0.002697 * Math.cos(6 * Math.PI * (dayOfYear - 1) / 365.24)
        + 0.00148 * Math.sin(6 * Math.PI * (dayOfYear - 1) / 365.24)) * TO_RAD;
        
    // Core of the sunrise/sunset formula to calculate the hour angle.
    const hourAngleArg = (Math.sin(altitude * TO_RAD) - Math.sin(lat * TO_RAD) * Math.sin(solarDeclination * TO_RAD)) / (Math.cos(lat * TO_RAD) * Math.cos(solarDeclination * TO_RAD));

    // Check for polar day (sun never sets) or polar night (sun never rises).
    if (hourAngleArg > 1) return { rise: null, set: null, type: 'polar_day' };
    if (hourAngleArg < -1) return { rise: null, set: null, type: 'polar_night' };

    const hourAngle = TO_DEG * Math.acos(hourAngleArg);
    
    // An approximation of the "Equation of Time", which accounts for the difference between solar time and clock time.
    const eqtime = 229.18 * (0.000075 + 0.001868 * Math.cos(2 * Math.PI * (dayOfYear - 1) / 365.24)
        - 0.032077 * Math.sin(2 * Math.PI * (dayOfYear - 1) / 365.24)
        - 0.014615 * Math.cos(4 * Math.PI * (dayOfYear - 1) / 365.24)
        - 0.040849 * Math.sin(4 * Math.PI * (dayOfYear - 1) / 365.24));
        
    // Calculate the time of solar noon in UTC (as a fraction of a day).
    const solarNoonUTC = (720 - 4 * lon - eqtime) / 1440;
    const halfDayMinutes = hourAngle * 4;
    
    // Calculate rise and set times in UTC (as a fraction of a day).
    const riseTimeUTC = (solarNoonUTC * 1440 - halfDayMinutes) / 1440;
    const setTimeUTC = (solarNoonUTC * 1440 + halfDayMinutes) / 1440;
    
    // Create Date objects from the calculated UTC fractions.
    const riseDate = new Date(date.getTime());
    riseDate.setUTCHours(0, 0, 0, 0);
    riseDate.setUTCMilliseconds(riseTimeUTC * 86400 * 1000);

    const setDate = new Date(date.getTime());
    setDate.setUTCHours(0, 0, 0, 0);
    setDate.setUTCMilliseconds(setTimeUTC * 86400 * 1000);

    return { rise: riseDate, set: setDate, type: 'normal' };
}

/**
 * Calculates the sun's altitude for a specific UTC time and geographical location.
 * @param {Date} date The date object (including time) for the calculation.
 * @param {number} lat Latitude in degrees.
 * @param {number} lon Longitude in degrees.
 * @returns {number} The sun's altitude in degrees above the horizon. Negative values indicate the sun is below the horizon.
 */
export function getSunAltitude(date, lat, lon) {
    // Calculate the day of the year (1-366).
    const dayOfYear = (Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()) - Date.UTC(date.getUTCFullYear(), 0, 0)) / 864e5;
    const latRad = lat * TO_RAD;

    // An approximation of the solar declination angle in radians.
    const solarDeclination = (0.006918 - 0.399912 * Math.cos(2 * Math.PI * (dayOfYear - 1) / 365.24)
        + 0.070257 * Math.sin(2 * Math.PI * (dayOfYear - 1) / 365.24)
        - 0.006758 * Math.cos(4 * Math.PI * (dayOfYear - 1) / 365.24)
        + 0.000907 * Math.sin(4 * Math.PI * (dayOfYear - 1) / 365.24)
        - 0.002697 * Math.cos(6 * Math.PI * (dayOfYear - 1) / 365.24)
        + 0.00148 * Math.sin(6 * Math.PI * (dayOfYear - 1) / 365.24));
        
    // An approximation of the "Equation of Time".
    const eqtime = 229.18 * (0.000075 + 0.001868 * Math.cos(2 * Math.PI * (dayOfYear - 1) / 365.24)
        - 0.032077 * Math.sin(2 * Math.PI * (dayOfYear - 1) / 365.24)
        - 0.014615 * Math.cos(4 * Math.PI * (dayOfYear - 1) / 365.24)
        - 0.040849 * Math.sin(4 * Math.PI * (dayOfYear - 1) / 365.24));
        
    // Calculate the hour angle from the true solar time.
    const timeOffset = eqtime + 4 * lon;
    const trueSolarTime = date.getUTCHours() * 60 + date.getUTCMinutes() + date.getUTCSeconds() / 60 + timeOffset;
    const hourAngle = (trueSolarTime / 4) - 180;
    const hourAngleRad = hourAngle * TO_RAD;
    
    // Final formula to calculate the sine of the sun's altitude.
    const sinAltitude = Math.sin(latRad) * Math.sin(solarDeclination) + Math.cos(latRad) * Math.cos(solarDeclination) * Math.cos(hourAngleRad);
    return TO_DEG * Math.asin(sinAltitude);
}

/**
 * Calculates a destination point given a starting point, distance, and initial bearing (azimuth).
 * This uses the Haversine formula on a spherical Earth model.
 * @param {number} lat Start latitude in degrees.
 * @param {number} lon Start longitude in degrees.
 * @param {number} distance Distance to travel in meters.
 * @param {number} bearing Initial bearing in degrees (0-360).
 * @returns {Array<number>} An array containing [latitude, longitude] of the destination point.
 */
export function destinationPoint(lat, lon, distance, bearing) {
    const R = 6371e3; // Earth's mean radius in meters
    const lat1 = lat * TO_RAD;
    const lon1 = lon * TO_RAD;
    const brng = bearing * TO_RAD;

    // Calculate the destination latitude.
    const lat2 = Math.asin(Math.sin(lat1) * Math.cos(distance / R) +
        Math.cos(lat1) * Math.sin(distance / R) * Math.cos(brng));
        
    // Calculate the destination longitude.
    const lon2 = lon1 + Math.atan2(Math.sin(brng) * Math.sin(distance / R) * Math.cos(lat1),
        Math.cos(distance / R) - Math.sin(lat1) * Math.sin(lat2));
        
    return [lat2 * TO_DEG, lon2 * TO_DEG];
}

/**
 * Calculates the initial bearing (forward azimuth) from a starting point to an end point.
 * @param {number} lat1 Latitude of the first point in degrees.
 * @param {number} lon1 Longitude of the first point in degrees.
 * @param {number} lat2 Latitude of the second point in degrees.
 * @param {number} lon2 Longitude of the second point in degrees.
 * @returns {number} The initial bearing in degrees (from 0 to 360).
 */
export function calculateBearing(lat1, lon1, lat2, lon2) {
    const y = Math.sin((lon2 - lon1) * TO_RAD) * Math.cos(lat2 * TO_RAD);
    const x = Math.cos(lat1 * TO_RAD) * Math.sin(lat2 * TO_RAD) -
              Math.sin(lat1 * TO_RAD) * Math.cos(lat2 * TO_RAD) * Math.cos((lon2 - lon1) * TO_RAD);
    const brng = Math.atan2(y, x) * TO_DEG;
    
    // Normalize the bearing to be within the range 0-360.
    return (brng + 360) % 360;
}
