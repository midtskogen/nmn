// --- Global Constants & Helpers ---

// Constant to convert degrees to radians.
export const TO_RAD = Math.PI / 180;
// Constant to convert radians to degrees.
export const TO_DEG = 180 / Math.PI;

/**
 * A utility function to create a DOM element with specified properties and attributes.
 * This simplifies the process of dynamically creating and configuring HTML elements.
 * @param {string} tag - The HTML tag for the element (e.g., 'div', 'button').
 * @param {object} [options={}] - An object where keys are element properties (like `className`, `textContent`, `onclick`)
 * or `dataset` for data-* attributes.
 * @returns {HTMLElement} The newly created and configured HTML element.
 */
export const createEl = (tag, options = {}) => {
    const el = document.createElement(tag);
    // Destructure the options to handle 'dataset' and 'style' as special cases.
    const { dataset, style, ...otherOptions } = options;
    // Assign all standard properties (like className, textContent, id, etc.) directly to the element.
    Object.assign(el, otherOptions);
    // Handle style: support both string (cssText) and object (individual properties).
    if (style) {
        if (typeof style === 'string') {
            el.style.cssText = style;
        } else {
            Object.assign(el.style, style);
        }
    }
    // If a `dataset` object was provided in the options, iterate over its keys
    // and assign them as `data-*` attributes on the element.
    if (dataset) {
        Object.assign(el.dataset, dataset);
    }

    return el;
};

/**
 * Checks if the browser likely supports HEVC (H.265) video playback.
 * It tests a comprehensive list of common and specific codec identifiers using
 * both `MediaSource.isTypeSupported` and `video.canPlayType` for broad compatibility.
 * @returns {boolean} Returns `true` if HEVC is likely supported, `false` otherwise.
 */
export const isHevcSupported = () => {
    // A list of MIME type strings with various HEVC codec profiles.
    // 'hvc1' and 'hev1' are different "boxes" for HEVC data in an MP4 container.
    const hevcMimeTypes = [
        'video/mp4; codecs="hvc1.1.6.L93.B0"', // More specific profile
        'video/mp4; codecs="hev1.1.6.L93.B0"',
        'video/mp4; codecs="hvc1"',             // Generic HEVC
        'video/mp4; codecs="hev1"'
    ];

    // `MediaSource.isTypeSupported` is the most reliable check for modern browsers that support Media Source Extensions.
    if (window.MediaSource && MediaSource.isTypeSupported) {
        for (const mimeType of hevcMimeTypes) {
            if (MediaSource.isTypeSupported(mimeType)) {
                return true;
            }
        }
    }

    // As a fallback, use the `canPlayType` method on a video element.
    // This method can return 'probably', 'maybe', or ''. Any non-empty string
    // indicates some level of potential support.
    const video = document.createElement('video');
    for (const mimeType of hevcMimeTypes) {
        if (video.canPlayType(mimeType)) {
            return true;
        }
    }

    // If none of the checks pass, assume no support.
    return false;
};

/**
 * Shows a temporary toast notification message.
 * @param {string} message - The message to display.
 * @param {string} [type='info'] - The type of toast ('info', 'warning', 'error').
 * @param {number} [duration=5000] - How long to show the toast in milliseconds.
 */
export function showToast(message, type = 'info', duration = 5000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: ${type === 'error' ? '#dc3545' : type === 'warning' ? '#ffc107' : '#17a2b8'};
        color: ${type === 'warning' ? '#000' : '#fff'};
        padding: 12px 24px;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        z-index: 10000;
        font-family: system-ui, sans-serif;
        font-size: 14px;
        max-width: 80%;
        text-align: center;
        opacity: 0;
        transition: opacity 0.3s ease;
    `;

    document.body.appendChild(toast);

    // Fade in
    requestAnimationFrame(() => {
        toast.style.opacity = '1';
    });

    // Remove after duration
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}
