import { getSunAltitude } from './calculations.js';

// --- Module-scoped variables ---
// These variables hold the Chart.js instance and the data used to create it,
// allowing them to be accessed and updated by different functions in this module.
let auroraChart = null;
let chartData = [];

/**
 * A helper function that determines the appropriate background color for a chart segment
 * based on the sun's altitude. It smoothly transitions from a day color to a night color
 * through the twilight period.
 * @param {number} altitude - The sun's altitude in degrees. Positive is day, negative is night.
 * @returns {string} An rgba color string to be used for styling.
 */
function getBackgroundColorForAltitude(altitude) {
    const dayColor = { r: 173, g: 216, b: 230 };   // Light blue for daytime
    const nightColor = { r: 10, g: 20, b: 40 };    // Dark blue for nighttime
    const opacity = 0.7;

    // Return full day or night color if the sun is above the horizon or below astronomical twilight.
    if (altitude >= 0) return `rgba(${dayColor.r}, ${dayColor.g}, ${dayColor.b}, ${opacity})`;
    if (altitude <= -18) return `rgba(${nightColor.r}, ${nightColor.g}, ${nightColor.b}, ${opacity})`;

    // For altitudes between 0 and -18 (twilight), calculate an interpolated color.
    const t = altitude / -18.0; // `t` will be a value from 0 to 1 representing the twilight progression.
    const r = Math.round(dayColor.r + t * (nightColor.r - dayColor.r));
    const g = Math.round(dayColor.g + t * (nightColor.g - dayColor.g));
    const b = Math.round(dayColor.b + t * (nightColor.b - dayColor.b));
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

/**
 * A custom Chart.js plugin designed to draw a background color behind each bar.
 * The color represents the day/twilight/night conditions for that time period,
 * creating a visual context for the aurora (Kp-index) data.
 */
const nightBackgroundPlugin = {
    id: 'nightBackground',
    // The `beforeDraw` hook ensures the background is drawn before the chart's main elements (like bars and grids).
    beforeDraw(chart, args, options) {
        // `options.colors` is a custom array of colors passed into the plugin's configuration.
        if (!options.colors || options.colors.length === 0) return;

        const { ctx, chartArea } = chart;
        // Iterate over each data point (and its corresponding label).
        chart.data.labels.forEach((label, index) => {
            // Get the metadata for the bar at this index.
            const bar = chart.getDatasetMeta(0).data[index];
            if (!bar) return; // Skip if the bar doesn't exist.

            ctx.save();
            ctx.fillStyle = options.colors[index]; // Use the pre-calculated color for this time slot.
            // Draw a rectangle that spans the full height of the chart area and has the width of the bar.
            ctx.fillRect(bar.x - bar.width / 2, chartArea.top, bar.width, chartArea.height);
            ctx.restore();
        });
    }
};

/**
 * Initializes the chart handler by registering the custom background plugin with Chart.js.
 * This only needs to be called once when the application starts.
 */
export function initChart() {
    Chart.register(nightBackgroundPlugin);
}

/**
 * Creates and renders the aurora activity (Kp-index) bar chart on a given canvas.
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D rendering context to draw the chart on.
 * @param {Array<object>} data - The formatted data for the chart, where each object has `label`, `value`, and `timestamp`.
 * @param {function} onBarClick - A callback function to execute when a bar in the chart is clicked.
 */
export function plotAuroraChart(ctx, data, onBarClick) {
    chartData = data; // Store data in the module scope for later access (e.g., for background updates).
    auroraChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.label),
            datasets: [{
                label: 'KP-indeks',
                data: data.map(d => d.value),
                // Dynamically color the bars based on the Kp-index value to indicate aurora strength.
                backgroundColor: data.map(d => d.value >= 5 ? 'rgba(255, 99, 132, 0.9)' : d.value >= 4 ? 'rgba(255, 206, 86, 0.9)' : 'rgba(75, 192, 192, 0.9)'),
                borderColor: 'rgba(100, 100, 100, 0.5)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            // Defines the click handler for the chart.
            onClick: (event) => {
                // Find which elements (bars) were clicked.
                const elements = auroraChart.getElementsAtEventForMode(event, 'index', { intersect: false }, true);
                if (elements.length > 0) {
                    const clickedDataPoint = data[elements[0].index];
                    if (clickedDataPoint) {
                        // Invoke the callback with the data corresponding to the clicked bar.
                        onBarClick(clickedDataPoint);
                    }
                }
            },
            plugins: {
                // Initialize the custom background plugin with an empty colors array. This will be populated later.
                nightBackground: { colors: [] },
                legend: { display: false }, // Hide the default chart legend.
                title: { display: true, text: 'Geomagnetisk aktivitet (KP-indeks, siste 7 dager, data fra NOAA)' },
                tooltip: {
                    callbacks: {
                        // Customize the tooltip title to show the full, precise timestamp.
                        title: (tooltipItems) => {
                            const dp = data[tooltipItems[0].dataIndex];
                            return dp ? new Date(dp.timestamp).toISOString().slice(0, 16).replace('T', ' ') : '';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    // The Kp-index scale is fixed from 0 to 9.
                    max: 9,
                    title: { display: true, text: 'KP-indeks' }
                }
            }
        }
    });
}

/**
 * Updates the background colors of the existing chart based on a given geographical location.
 * This is called when the user selects a station or moves the map, allowing the day/night
 * visualization to be specific to that location.
 * @param {number} lat - The latitude for the sun position calculation.
 * @param {number} lon - The longitude for the sun position calculation.
 */
export function updateChartBackground(lat, lon) {
    if (!auroraChart || chartData.length === 0) return;

    // For each data point in the chart, calculate the sun's altitude at that time and location.
    // Then, determine the corresponding background color.
    auroraChart.options.plugins.nightBackground.colors = chartData.map(dataPoint =>
        getBackgroundColorForAltitude(getSunAltitude(new Date(dataPoint.timestamp), lat, lon))
    );
    
    // Trigger a chart update to redraw it with the new background colors.
    auroraChart.update();
}
