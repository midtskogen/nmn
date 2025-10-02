import { getSunAltitude } from './calculations.js';

// --- Module-scoped variables ---
let auroraChart = null;
let chartData = [];
let t = (key) => key; // The translation function, initialized to a fallback.

/**
 * A helper function that determines the appropriate background color for a chart segment
 * based on the sun's altitude. It smoothly transitions from a day color to a night color
 * through the twilight period.
 * @param {number} altitude - The sun's altitude in degrees. Positive is day, negative is night.
 * @returns {string} An rgba color string to be used for styling.
 */
function getBackgroundColorForAltitude(altitude) {
    const dayColor = { r: 173, g: 216, b: 230 }; // Light blue for daytime
    const nightColor = { r: 10, g: 20, b: 40 }; // Dark blue for nighttime
    const opacity = 0.7;
    
    if (altitude >= 0) return `rgba(${dayColor.r}, ${dayColor.g}, ${dayColor.b}, ${opacity})`;
    if (altitude <= -18) return `rgba(${nightColor.r}, ${nightColor.g}, ${nightColor.b}, ${opacity})`;
    
    const trans = altitude / -18.0;
    const r = Math.round(dayColor.r + trans * (nightColor.r - dayColor.r));
    const g = Math.round(dayColor.g + trans * (nightColor.g - dayColor.g));
    const b = Math.round(dayColor.b + trans * (nightColor.b - dayColor.b));
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

/**
 * A custom Chart.js plugin designed to draw a background color behind each bar.
 * The color represents the day/twilight/night conditions for that time period,
 * creating a visual context for the aurora (Kp-index) data.
 */
const nightBackgroundPlugin = {
    id: 'nightBackground',
    beforeDraw(chart, args, options) {
        if (!options.colors || options.colors.length === 0) return;

        const { ctx, chartArea } = chart;
        chart.data.labels.forEach((label, index) => {
            const bar = chart.getDatasetMeta(0).data[index];
            if (!bar) return;

            ctx.save();
            ctx.fillStyle = options.colors[index];
            ctx.fillRect(bar.x - bar.width / 2, chartArea.top, bar.width, chartArea.height);
            ctx.restore();
        });
    }
};

/**
 * Initializes the chart handler by registering the custom background plugin and storing the translation function.
 * @param {function} translationFunc - The translation function from main.js.
 */
export function initChart(translationFunc) {
    t = translationFunc;
    Chart.register(nightBackgroundPlugin);
}

/**
 * Creates and renders the aurora activity (Kp-index) bar chart on a given canvas.
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D rendering context to draw the chart on.
 * @param {Array<object>} data - The formatted data for the chart, where each object has `label`, `value`, and `timestamp`.
 * @param {function} onBarClick - A callback function to execute when a bar in the chart is clicked.
 */
export function plotAuroraChart(ctx, data, onBarClick) {
    chartData = data;
    auroraChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.label),
            datasets: [{
                label: t('chart_kp_label'),
                data: data.map(d => d.value),
                backgroundColor: data.map(d => d.value >= 5 ? 'rgba(255, 99, 132, 0.9)' : d.value >= 4 ? 'rgba(255, 206, 86, 0.9)' : 'rgba(75, 192, 192, 0.9)'),
                borderColor: 'rgba(100, 100, 100, 0.5)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            onClick: (event) => {
                const elements = auroraChart.getElementsAtEventForMode(event, 'index', { intersect: false }, true);
                if (elements.length > 0) {
                    const clickedDataPoint = data[elements[0].index];
                    if (clickedDataPoint) {
                        onBarClick(clickedDataPoint);
                    }
                }
            },
            plugins: {
                nightBackground: { colors: [] },
                legend: { display: false },
                title: { display: true, text: t('chart_title') },
                tooltip: {
                    callbacks: {
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
                    max: 9,
                    title: { display: true, text: t('chart_kp_axis_label') }
                }
            }
        }
    });
}

/**
 * Updates the background colors of the existing chart based on a given geographical location.
 * @param {number} lat - The latitude for the sun position calculation.
 * @param {number} lon - The longitude for the sun position calculation.
 */
export function updateChartBackground(lat, lon) {
    if (!auroraChart || chartData.length === 0) return;
    
    auroraChart.options.plugins.nightBackground.colors = chartData.map(dataPoint =>
        getBackgroundColorForAltitude(getSunAltitude(new Date(dataPoint.timestamp), lat, lon))
    );
    auroraChart.update();
}
