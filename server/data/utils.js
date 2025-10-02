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
    // Destructure the options to handle 'dataset' as a special case, separating it
    // from other standard element properties.
    const { dataset, ...otherOptions } = options;
    // Assign all standard properties (like className, textContent, id, etc.) directly to the element.
    Object.assign(el, otherOptions);
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
