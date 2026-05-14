<!DOCTYPE html>
<html lang="no">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meteorobservasjoner - Velg hendelser</title>
    <style>
        /* Base styles */
        body { background: #ffffff; color: #000018; font-family: Verdana, Geneva, Arial, Helvetica, sans-serif; font-size: 12px; }
        h1 { text-align: center; color: #082060; }
        h2 { text-align: center; }
        table { width: 100%; border-collapse: collapse; }
        td { vertical-align: top; }
        p { text-align: justify; }
        a { text-decoration: none; }
        hr { margin: 20px 0; }
        #observation-table-info { margin-bottom: 20px; }
        #observation-table-info a:link { color: #c01010; }
        #observation-table-info a:visited { color: #601010; }

        /* Responsive layout */
        .info-container { display: flex; align-items: flex-start; gap: 20px; padding: 0 15px; }
        .info-text { flex: 1; min-width: 60%; }
        .info-image { flex-shrink: 0; text-align: center; }
        .info-image img { max-width: 100%; height: auto; }
        @media screen and (max-width: 768px) { .info-container { flex-direction: column; } }

        /* Wrappers */
        .content-wrapper { padding: 0 15px; }
        .scrollable-table-container { overflow-x: auto; -webkit-overflow-scrolling: touch; }
        .scrollable-table-container img { width: 256px; height: auto; }
	.scrollable-table-container table { width: auto; min-width: 100%; }

        /* Form and results styles */
        .selection-form { padding: 0 15px; margin-bottom: 15px; }
        .submit-button { background-color: #082060; color: white; font-weight: bold; text-align: center; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; font-size: 1.1em; width: 100%; margin-bottom: 15px; }
        .submit-button:hover { background-color: #0a287a; }
        #selection-results { display: none; padding: 15px; border: 1px solid #082060; border-radius: 5px; margin: 20px 0; background-color: #f5f8fa; }
        
        /* Archive section styles */
        .archive-section button { background-color: #f0f0f0; border: 1px solid #ccc; padding: 8px 12px; width: 100%; text-align: left; cursor: pointer; font-size: 1.1em; margin-top: 5px; margin-bottom: 5px; border-radius: 4px; }
        .archive-section button:hover { background-color: #e0e0e0; }
        .spoiler-content { border: 1px solid #ccc; padding: 15px; margin-top: -5px; border-top: none; border-radius: 0 0 4px 4px; overflow-x: auto; }
        
        /* Styles for dynamically loaded content */
        .months-table { border: 1px solid #ddd; }
        .month-cell { vertical-align: top; text-align: center; border: 1px solid #ddd; padding: 8px; }
        .month-header { font-weight: bold; border-bottom: 2px solid #000; margin-bottom: 5px; display: block; cursor: pointer; white-space: nowrap; }
        .month-header:hover { background-color: #f0f0f0; }
        .event-container { display: flex; align-items: center; margin-bottom: 5px; padding-bottom: 5px; border-bottom: 1px solid #eee; }
        .event-container:last-child { border-bottom: none; }
        .observation-link { display: block; }
        .observation-link span { display: block; }
        .location-details { font-size: 0.8em; }
        .media-container { margin-top: 5px; }
        .media-swap-container { display: flex; align-items: center; gap: 5px; }
        .media-swap-container img, .media-swap-container video { margin: 2px; border: 1px solid #ccc; width: 256px; height: auto; vertical-align: top; }
        /* MODIFIED: Override style to show image checkboxes when viewed from id.php */
        .image-checkbox { display: inline-block; }
        .observation-link.low-altitude, .observation-link.low-altitude:visited { color: red; font-weight: bold; }
        .observation-link.normal-altitude, .observation-link.normal-altitude:visited { color: black; font-weight: bold; }
        .observation-link.unlikely-altitude, .observation-link.unlikely-altitude:visited { color: grey; }
        .observation-link.unknown-altitude, .observation-link.unknown-altitude:visited { color: black; font-weight: normal; }
        .filter-container { text-align: center; margin-bottom: 20px; }
        .filter-container fieldset { border: 1px solid #ccc; padding: 10px; margin: 10px; display: inline-block; vertical-align: top; }
        .filter-container legend { font-weight: bold; }
        .filter-container label { margin: 0 10px; display: block; text-align: left; }
    </style>
</head>
<body>

    <h1>De siste registreringene</h1>

    <form id="event-selection-form" class="selection-form" onsubmit="showSelections(event)">
        <button type="submit" class="submit-button">Se avkrysninger</button>

        <div id="selection-results"></div>

        <div class="content-wrapper">
            <div class="scrollable-table-container" id="main-table-container">
                <?php
                    // --- FIX ---
                    // Load the static file content but strip its script to prevent conflicts
                    $static_content = file_get_contents('index-static_nb.html');
                    $static_content_no_script = preg_replace('/<script\b[^>]*>[\s\S]*?<\/script>/im', '', $static_content);
                    echo $static_content_no_script;
                    // --- END FIX ---
                ?>
            </div>
        </div>
        <hr>
        <div class="archive-section">
            <h2>Arkiv</h2>
            <p>Velg et år for å laste inn eldre hendelser.</p>
            <?php
            $currentYear = date('Y');
            for ($year = $currentYear; $year >= 2015; $year--) {
                // Button and container for July-December
                echo '<button title="Click to show/hide content" type="button" onclick="toggleSpoiler(\'' . $year . '_b\')">Vis/skjul ' . $year . ' (Juli - Desember)</button>';
                echo '<div id="spoiler' . $year . '_b" class="spoiler-content" style="display:none;"></div>';
                // Button and container for January-June
                echo '<button title="Click to show/hide content" type="button" onclick="toggleSpoiler(\'' . $year . '_a\')">Vis/skjul ' . $year . ' (Januar - Juni)</button>';
                echo '<div id="spoiler' . $year . '_a" class="spoiler-content" style="display:none;"></div>';
            }
            ?>
        </div>
    </form>

    <script>
    /**
     * MODIFIED FUNCTION
     * Adds checkboxes for events and individual detections.
     * Part 2 now includes a fallback to read 'img.src' if 'data-src'
     * was already removed by a conflicting lazy-load script.
     */
    function addCheckboxes(containerElement) {
        if (!containerElement) return;

        // Part 1: Add checkboxes for main events
        const eventContainers = containerElement.querySelectorAll('.event-container');
        eventContainers.forEach(container => {
            const link = container.querySelector('a.observation-link');
            const existingCheckbox = container.querySelector(':scope > input[type="checkbox"]');
            if (link && !existingCheckbox) {
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.name = 'events[]';
                const href = link.getAttribute('href'); // e.g., /meteor/./YYYYMMDD/hhmmss/
                if (href) {
                    let path = href.replace('/meteor/', '').replace(/\/$/, ''); // e.g., ./YYYYMMDD/hhmmss
                    if (path.startsWith('./')) {
                        path = path.substring(2); // e.g., YYYYMMDD/hhmmss
                    }
                    checkbox.value = path;
                }
                container.insertBefore(checkbox, link);
            }
        });

        // Part 2: Add checkboxes for individual detections (images)
        const mediaSwapContainers = containerElement.querySelectorAll('.media-swap-container');
        mediaSwapContainers.forEach(mediaDiv => {
            const img = mediaDiv.querySelector('img');
            const existingCheckbox = mediaDiv.querySelector(':scope > input[type="checkbox"]');
            
            if (img && !existingCheckbox) {
                // --- FIX ---
                // Check for data-src first. If it's gone (due to race condition),
                // fall back to the actual img.src, but only if it's not the placeholder.
                const dataSrc = img.getAttribute('data-src');
                const currentSrc = img.src;
                const placeholderSrc = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP'; // Partial match
                
                let pathSource = null;

                if (dataSrc) {
                    pathSource = dataSrc; // Best case: data-src is still there
                } else if (currentSrc && !currentSrc.startsWith(placeholderSrc)) {
                    pathSource = currentSrc; // Fallback: image was lazy-loaded, use its src
                }
                // --- END FIX ---

                if (pathSource) {
                    // pathSource is e.g., /meteor/./YYYYMMDD/hhmmss/station/camX/file.jpg
                    let path = pathSource.replace('/meteor/', ''); // e.g., ./YYYYMMDD/hhmmss/station/camX/file.jpg
                    if (path.startsWith('./')) {
                        path = path.substring(2); // e.g., YYYYMMDD/hhmmss/station/camX/file.jpg
                    }
                    path = path.substring(0, path.lastIndexOf('/')); // e.g., YYYYMMDD/hhmmss/station/camX

                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.name = 'events[]';
                    checkbox.value = path;
                    checkbox.className = 'image-checkbox';
                    
                    mediaDiv.insertBefore(checkbox, img);
                }
            }
        });
    }
    
    function enhanceFilterControls(filterContainer) {
        const multiStationRadio = filterContainer.querySelector('input[name="eventFilter"][value="multi-station"]');
        if (!multiStationRadio) return;
        multiStationRadio.checked = false;
        const allRadio = filterContainer.querySelector('input[name="eventFilter"][value="all"]');
        if (allRadio) {
            allRadio.checked = true;
        }
        const singleStationLabel = document.createElement('label');
        const singleStationRadio = document.createElement('input');
        singleStationRadio.type = 'radio';
        singleStationRadio.name = 'eventFilter';
        singleStationRadio.value = 'single-station';
        singleStationRadio.checked = false;
        singleStationLabel.appendChild(singleStationRadio);
        singleStationLabel.appendChild(document.createTextNode('Vis bare enkeltstasjonhendelser'));
        multiStationRadio.parentElement.insertAdjacentElement('afterend', singleStationLabel);
    }

    /**
     * Fetches the correct language-specific archive file (e.g., YYYY_nb_a.html).
     * Runs addCheckboxes BEFORE lazy-loading images.
     */
    function toggleSpoiler(archiveId) {
        const container = document.getElementById('spoiler' + archiveId);
        if (!container) return;
        if (container.style.display === 'none') {
            container.style.display = 'block';
            if (!container.hasAttribute('data-loaded')) {
                container.innerHTML = '<p>Laster inn data...</p>';

                const parts = archiveId.split('_'); // ['YYYY', 'a']
                const lang = 'nb'; // Hardcoded for this Norwegian page
                const fileName = `${parts[0]}_${lang}_${parts[1]}.html`; // 'YYYY_nb_a.html'
                const fetchUrl = '/meteor/' + fileName;

                fetch(fetchUrl)
                    .then(response => {
                        if (!response.ok) throw new Error('Network response was not ok. URL: ' + fetchUrl);
                        return response.text();
                    })
                    .then(html => {
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(html, 'text/html');
                        doc.querySelector('script')?.remove(); // Remove script from loaded content
                        container.innerHTML = doc.body.innerHTML;

                        // Run addCheckboxes FIRST, while img[data-src] still exists.
                        addCheckboxes(container);

                        // Now, lazy-load the images for the archive content
                        const imagesInContainer = container.querySelectorAll('img[data-src]');
                        imagesInContainer.forEach(img => {
                            const src = img.getAttribute('data-src');
                            if (src) {
                                img.src = src;
                                img.removeAttribute('data-src');
                            }
                        });

                        // Continue with setup
                        const filterControls = container.querySelector('.filter-container');
                        if (filterControls) enhanceFilterControls(filterControls);
                        applyFilters();
                        const unprocessedRadio = container.querySelector('input[name="displayChoice"][value="unprocessed"]');
                        if (unprocessedRadio) {
                            unprocessedRadio.checked = true;
                            applyFilters();
                        }
                        container.setAttribute('data-loaded', 'true');
                    })
                    .catch(error => {
                        container.innerHTML = '<p style="color:red;">Kunne ikke laste inn data. Filen finnes kanskje ikke.</p>';
                        console.error('Fetch error:', error);
                    });
            }
        } else {
            container.style.display = 'none';
        }
    }

    function applyFilters() {
        const form = document.getElementById('event-selection-form');
        const selectedEventFilter = form.querySelector('input[name="eventFilter"]:checked')?.value;
        const selectedDisplayChoice = form.querySelector('input[name="displayChoice"]:checked')?.value;
        if (!selectedEventFilter || !selectedDisplayChoice) return;

        const eventContainers = form.querySelectorAll('.event-container');
        eventContainers.forEach(container => {
            const eventType = container.getAttribute('data-event-type');
            const showerType = container.getAttribute('data-shower');
            let showByEventType = false;
            const isMultiStation = eventType === 'multi-station-normal' || eventType === 'meteorite-candidate';
            switch (selectedEventFilter) {
                case 'all': showByEventType = true; break;
                case 'multi-station': showByEventType = isMultiStation; break;
                case 'single-station': showByEventType = eventType === 'single-station'; break;
                case 'candidates': showByEventType = eventType === 'meteorite-candidate'; break;
                case 'perseider': case 'sorlige-taurider': case 'nordlige-taurider': case 'leonider': case 'geminider':
                    showByEventType = isMultiStation && showerType === selectedEventFilter;
                    break;
            }
            container.style.display = showByEventType ? 'flex' : 'none';
        });

        const mediaContainers = form.querySelectorAll('.media-container');
        mediaContainers.forEach(c => c.style.display = 'none');
        if (selectedDisplayChoice !== 'none') {
            const targetClass = { 'processed': '.processed-images', 'unprocessed': '.unprocessed-images' }[selectedDisplayChoice];
            if (targetClass) {
                const containersToShow = form.querySelectorAll(targetClass);
                containersToShow.forEach(c => {
                    if (c.closest('.event-container').style.display === 'flex') {
                        c.style.display = 'block';
                    }
                });
            }
        }
    }

    function copySelectionsToClipboard() {
        const textToCopy = document.getElementById('selection-text')?.textContent;
        const copyIcon = document.getElementById('copy-icon');
        if (!textToCopy || !copyIcon) return;

        navigator.clipboard.writeText(textToCopy).then(() => {
            const originalTitle = copyIcon.getAttribute('title');
            copyIcon.setAttribute('title', 'Kopiert!');
            setTimeout(() => {
                copyIcon.setAttribute('title', originalTitle);
            }, 2000); // Reset tooltip after 2 seconds
        }).catch(err => {
            console.error('Kunne ikke kopiere: ', err);
            alert('Kunne ikke kopiere til utklippstavlen.');
        });
    }

    function showSelections(event) {
        event.preventDefault(); 
        
        const form = document.getElementById('event-selection-form');
        const checkedBoxes = form.querySelectorAll('input[type="checkbox"]:checked');
        const resultsContainer = document.getElementById('selection-results');
        
        if (checkedBoxes.length === 0) {
            resultsContainer.innerHTML = '<h2>Valgte hendelser:</h2><p style="color: #c01010;">Vennligst velg minst én hendelse.</p>';
            resultsContainer.style.display = 'block';
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
            return;
        }

        const paths = Array.from(checkedBoxes).map(box => box.value);
        const pathsString = paths.join(' ');

        const copySvg = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16" style="vertical-align: middle;">
          <path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/>
          <path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/>
        </svg>`;

        const outputHtml = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <h2>Valgte hendelser:</h2>
                <span id="copy-icon" title="Kopier til utklippstavle" style="cursor: pointer; display: inline-flex; align-items: center;" onclick="copySelectionsToClipboard()">
                    ${copySvg}
                </span>
            </div>
            <p id="selection-text" style="font-family: monospace; word-break: break-all; margin-top: 5px; text-align: left;">${pathsString}</p>`;

        resultsContainer.innerHTML = outputHtml;
        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    document.addEventListener('DOMContentLoaded', function() {
        const mainTableContainer = document.getElementById('main-table-container');
        
        // --- FIX ---
        // Manually lazy-load images for the static content *after* adding checkboxes
        
        // 1. Add checkboxes while data-src still exists
        addCheckboxes(mainTableContainer);
        
        // 2. Now, lazy-load the static images
        const staticImages = mainTableContainer.querySelectorAll('img[data-src]');
        staticImages.forEach(img => {
            const src = img.getAttribute('data-src');
            if (src) {
                img.src = src;
                img.removeAttribute('data-src');
            }
        });
        // --- END FIX ---


        enhanceFilterControls(mainTableContainer);
        const mainUnprocessedRadio = mainTableContainer.querySelector('input[name="displayChoice"][value="unprocessed"]');
        if (mainUnprocessedRadio) mainUnprocessedRadio.checked = true;
        applyFilters();
        const form = document.getElementById('event-selection-form');
        form.addEventListener('change', (e) => {
            if (e.target.name === 'eventFilter' || e.target.name === 'displayChoice') {
                applyFilters();
            }
        });

        // This video-swap logic works for the whole page,
        // including content loaded later into spoiler divs.
        document.body.addEventListener('mouseover', function(e) {
            const container = e.target.closest('.media-swap-container');
            if (container) {
                const img = container.querySelector('img');
                let video = container.querySelector('video');
                // Check for data-videosrc on the img tag
                const videoSrc = img ? img.getAttribute('data-videosrc') : null;

                if (!video && videoSrc) {
                    video = document.createElement('video');
                    video.src = videoSrc;
                    video.style.width = img.clientWidth > 0 ? img.clientWidth + 'px' : '256px';
                    video.style.height = img.clientHeight > 0 ? img.clientHeight + 'px' : 'auto';
                    video.loop = true;
                    video.muted = true;
                    video.playsInline = true;
                    video.style.display = 'none';
                    container.appendChild(video);
                }
                if (img && video) {
                    img.style.display = 'none';
                    video.style.display = 'inline-block';
                    video.play().catch(error => {});
                }
            }
        });

        document.body.addEventListener('mouseout', function(e) {
            const container = e.target.closest('.media-swap-container');
            if (container) {
                const img = container.querySelector('img');
                const video = container.querySelector('video');
                if (img && video) {
                    video.pause();
                    video.style.display = 'none';
                    img.style.display = 'inline-block';
                }
            }
        });

    });
    </script>
</body>
</html>
