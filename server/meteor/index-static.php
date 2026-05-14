<?php
/**
 * Main script to generate static, multilingual HTML pages for meteor observations.
 *
 * This script is intended to be run via cron. It generates a separate HTML file
 * for each supported language, containing the observation data for the last 6 months.
 *
 * It can also be run in a secondary mode from the CLI with a year argument
 * (e.g., `php index-static.php 2024`) to generate multilingual yearly archive files.
 */

// --- Configuration & Setup ---
define('BASE_PATH', '/home/httpd/norskmeteornettverk.no/meteor');
chdir(BASE_PATH);

$isCliMode = php_sapi_name() == 'cli';
$isYearArgProvided = isset($argv[1]);

// --- Translations ---
$translations = [
    'nb_NO' => [
        'lang_short' => 'nb',
        'lang_name' => 'Norsk',
        'archive_title_for_year' => 'Observasjoner for %d',
        'archive_part1_title' => '(januar - juni)',
        'archive_part2_title' => '(juli - desember)',
        'archive_back_link_text' => 'Tilbake til de siste registreringene',
        'filter_events_header' => 'Filtrer hendelser',
        'filter_show_all' => 'Vis alle',
        'filter_multi_station' => 'Flerstasjonshendelser',
        'filter_candidates' => 'Meteorittkandidater',
        'filter_perseids' => 'Perseider',
        'filter_southern_taurids' => 'Sørlige taurider',
        'filter_northern_taurids' => 'Nordlige taurider',
        'filter_leonids' => 'Leonider',
        'filter_geminids' => 'Geminider',
        'display_options_header' => 'Visningsvalg',
        'display_no_images' => 'Ingen bilder',
        'display_unprocessed' => 'Vis uprosseserte bilder',
        'display_processed' => 'Vis prosseserte bilder',
        'last_6_months' => 'Siste 6 måneder',
        'hidden_singular' => 'skjult',
        'hidden_plural' => 'skjulte',
        'month_names' => ["01" => "Januar", "02" => "Februar", "03" => "Mars", "04" => "April", "05" => "Mai", "06" => "Juni", "07" => "Juli", "08" => "August", "09" => "September", "10" => "Oktober", "11" => "November", "12" => "Desember"],
    ],
    'en_GB' => [
        'lang_short' => 'en',
        'lang_name' => 'English',
        'archive_title_for_year' => 'Detections for %d',
        'archive_part1_title' => '(January - June)',
        'archive_part2_title' => '(July - December)',
        'archive_back_link_text' => 'Back to the latest detections',
        'filter_events_header' => 'Filter Events',
        'filter_show_all' => 'Show all',
        'filter_multi_station' => 'Multi-station events',
        'filter_candidates' => 'Meteorite candidates',
        'filter_perseids' => 'Perseids',
        'filter_southern_taurids' => 'Southern Taurids',
        'filter_northern_taurids' => 'Northern Taurids',
        'filter_leonids' => 'Leonids',
        'filter_geminids' => 'Geminids',
        'display_options_header' => 'Display Options',
        'display_no_images' => 'No images',
        'display_unprocessed' => 'Show unprocessed images',
        'display_processed' => 'Show processed images',
        'last_6_months' => 'Last 6 months',
        'hidden_singular' => 'hidden',
        'hidden_plural' => 'hidden',
        'month_names' => ["01" => "January", "02" => "February", "03" => "March", "04" => "April", "05" => "May", "06" => "June", "07" => "July", "08" => "August", "09" => "September", "10" => "October", "11" => "November", "12" => "December"],
    ],
    'de_DE' => [
        'lang_short' => 'de',
        'lang_name' => 'Deutsch',
        'archive_title_for_year' => 'Beobachtungen für %d',
        'archive_part1_title' => '(Januar - Juni)',
        'archive_part2_title' => '(Juli - Dezember)',
        'archive_back_link_text' => 'Zurück zu den neuesten Erfassungen',
        'filter_events_header' => 'Ereignisse filtern',
        'filter_show_all' => 'Alle anzeigen',
        'filter_multi_station' => 'Mehrstationsereignisse',
        'filter_candidates' => 'Meteoritenkandidaten',
        'filter_perseids' => 'Perseiden',
        'filter_southern_taurids' => 'Südliche Tauriden',
        'filter_northern_taurids' => 'Nördliche Tauriden',
        'filter_leonids' => 'Leoniden',
        'filter_geminids' => 'Geminiden',
        'display_options_header' => 'Anzeigeoptionen',
        'display_no_images' => 'Keine Bilder',
        'display_unprocessed' => 'Unverarbeitete Bilder anzeigen',
        'display_processed' => 'Verarbeitete Bilder anzeigen',
        'last_6_months' => 'Letzte 6 Monate',
        'hidden_singular' => 'versteckt',
        'hidden_plural' => 'versteckt',
        'month_names' => ["01" => "Januar", "02" => "Februar", "03" => "März", "04" => "April", "05" => "Mai", "06" => "Juni", "07" => "Juli", "08" => "August", "09" => "September", "10" => "Oktober", "11" => "November", "12" => "Dezember"],
    ],
    'cs_CZ' => [
        'lang_short' => 'cs',
        'lang_name' => 'Čeština',
        'archive_title_for_year' => 'Pozorování za rok %d',
        'archive_part1_title' => '(leden - červen)',
        'archive_part2_title' => '(červenec - prosinec)',
        'archive_back_link_text' => 'Zpět na nejnovější záznamy',
        'filter_events_header' => 'Filtrovat události',
        'filter_show_all' => 'Zobrazit vše',
        'filter_multi_station' => 'Vícestaniční události',
        'filter_candidates' => 'Kandidáti na meteority',
        'filter_perseids' => 'Perseidy',
        'filter_southern_taurids' => 'Jižní Tauridy',
        'filter_northern_taurids' => 'Severní Tauridy',
        'filter_leonids' => 'Leonidy',
        'filter_geminids' => 'Geminidy',
        'display_options_header' => 'Možnosti zobrazení',
        'display_no_images' => 'Žádné obrázky',
        'display_unprocessed' => 'Zobrazit nezpracované obrázky',
        'display_processed' => 'Zobrazit zpracované obrázky',
        'last_6_months' => 'Posledních 6 měsíců',
        'hidden_singular' => 'skrytý',
        'hidden_plural' => 'skryto',
        'month_names' => ["01" => "Leden", "02" => "Únor", "03" => "Březen", "04" => "Duben", "05" => "Květen", "06" => "Červen", "07" => "Červenec", "08" => "Srpen", "09" => "Září", "10" => "Říjen", "11" => "Listopad", "12" => "Prosinec"],
    ],
    'fi_FI' => [
        'lang_short' => 'fi',
        'lang_name' => 'Suomi',
        'archive_title_for_year' => 'Havainnot vuodelta %d',
        'archive_part1_title' => '(tammi - kesäkuu)',
        'archive_part2_title' => '(heinä - joulukuu)',
        'archive_back_link_text' => 'Takaisin viimeisimpiin havaintoihin',
        'filter_events_header' => 'Suodata tapahtumia',
        'filter_show_all' => 'Näytä kaikki',
        'filter_multi_station' => 'Moniasemahavainnot',
        'filter_candidates' => 'Meteoriittiehdokkaat',
        'filter_perseids' => 'Perseidit',
        'filter_southern_taurids' => 'Eteläiset tauridit',
        'filter_northern_taurids' => 'Pohjoiset tauridit',
        'filter_leonids' => 'Leonidit',
        'filter_geminids' => 'Geminidit',
        'display_options_header' => 'Näyttöasetukset',
        'display_no_images' => 'Ei kuvia',
        'display_unprocessed' => 'Näytä käsittelemättömät kuvat',
        'display_processed' => 'Näytä käsitellyt kuvat',
        'last_6_months' => 'Viimeiset 6 kuukautta',
        'hidden_singular' => 'piilotettu',
        'hidden_plural' => 'piilotettua', // *** KORRIGERT HER ***
        'month_names' => ["01" => "Tammikuu", "02" => "Helmikuu", "03" => "Maaliskuu", "04" => "Huhtikuu", "05" => "Toukokuu", "06" => "Kesäkuu", "07" => "Heinäkuu", "08" => "Elokuu", "09" => "Syyskuu", "10" => "Lokakuu", "11" => "Marraskuu", "12" => "Joulukuu"],
    ],
];

// --- Function Definitions ---

/**
 * Gathers and organizes event data from the filesystem for specific years.
 * @param array $yearsToScan An array of years to gather data for.
 * @return array A structured array of event paths, organized by year and month.
 */
function gatherEventData($yearsToScan) {
    $datesByYearMonth = [];
    foreach ($yearsToScan as $year) {
        $dateDirs = glob("./{$year}*", GLOB_ONLYDIR);
        if (!$dateDirs) continue;
        rsort($dateDirs);

        foreach ($dateDirs as $dateDir) {
            $timeDirs = glob("{$dateDir}/[0-9][0-9][0-9][0-9][0-9][0-9]", GLOB_ONLYDIR);
            foreach ($timeDirs as $timeDir) {
                if (!file_exists("{$timeDir}/index.php")) {
                    continue;
                }

                // Exclude pictureless observations (e.g. only obs_*.txt without generated images).
                // A valid event must have at least one expected media artifact.
                $hasFireballImages = !empty(glob("{$timeDir}/*/*/fireball*.jpg"));
                $hasMultiStationMedia = file_exists("{$timeDir}/map.jpg") || !empty(glob("{$timeDir}/*_map.jpg"));
                if (!$hasFireballImages && !$hasMultiStationMedia) {
                    continue;
                }

                $month = substr(basename($dateDir), 4, 2);
                $datesByYearMonth[$year][$month][] = $timeDir;
            }
        }
    }
    foreach ($datesByYearMonth as &$months) {
        if(is_array($months)){
            foreach ($months as &$events) {
                if(is_array($events)) rsort($events);
            }
        }
    }
    return $datesByYearMonth;
}

/**
 * Formats an English day of the month with the correct ordinal suffix (st, nd, rd, th).
 * @param int $day The day number.
 * @return string The formatted day string.
 */
function formatDayWithSuffix($day) {
    if (!in_array(($day % 100), [11, 12, 13])) {
        switch ($day % 10) {
            case 1: return $day . 'st';
            case 2: return $day . 'nd';
            case 3: return $day . 'rd';
        }
    }
    return $day . 'th';
}

/**
 * Generates the HTML for a media item (image and potential video).
 * @param string $imgPath Path to the image file.
 * @param string $videoExt The video file extension to look for.
 * @param string $altText The alt text for the image.
 * @return string The generated HTML.
 */
function generateMediaItem($imgPath, $videoExt, $altText) {
    $basePath = str_replace(['_orig.jpg', '.jpg'], '', $imgPath);
    $videoFullPath = $basePath . $videoExt;
    $webVideoPath = '/meteor/' . $videoFullPath;
    $webImagePath = '/meteor/' . $imgPath;
    $videoDataAttr = file_exists($videoFullPath) ? "data-videosrc='{$webVideoPath}'" : '';
    $placeholderSrc = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
    $imageHTML = "<img src='{$placeholderSrc}' data-src='{$webImagePath}' width='256' loading='lazy' alt='{$altText}' {$videoDataAttr}>";
    return "<div class='media-swap-container'>{$imageHTML}</div>";
}

/**
 * Generates the HTML for the interactive filter controls.
 * @param array $t The translation array for the current language.
 * @return string The HTML for the filters.
 */
function generateFilterControls($t) {
    return <<<HTML
    <div class="filter-container">
        <fieldset>
            <legend>{$t['filter_events_header']}</legend>
            <label><input type="radio" name="eventFilter" value="all">{$t['filter_show_all']}</label>
            <label><input type="radio" name="eventFilter" value="multi-station" checked>{$t['filter_multi_station']}</label>
            <label><input type="radio" name="eventFilter" value="candidates">{$t['filter_candidates']}</label>
            <label><input type="radio" name="eventFilter" value="perseider">{$t['filter_perseids']}</label>
            <label><input type="radio" name="eventFilter" value="sorlige-taurider">{$t['filter_southern_taurids']}</label>
            <label><input type="radio" name="eventFilter" value="nordlige-taurider">{$t['filter_northern_taurids']}</label>
            <label><input type="radio" name="eventFilter" value="leonider">{$t['filter_leonids']}</label>
            <label><input type="radio" name="eventFilter" value="geminider">{$t['filter_geminids']}</label>
        </fieldset>
        <fieldset>
            <legend>{$t['display_options_header']}</legend>
            <label><input type="radio" name="displayChoice" value="none" checked>{$t['display_no_images']}</label>
            <label><input type="radio" name="displayChoice" value="unprocessed">{$t['display_unprocessed']}</label>
            <label><input type="radio" name="displayChoice" value="processed">{$t['display_processed']}</label>
        </fieldset>
    </div>
HTML;
}

/**
 * Generates the HTML for the main observation table.
 * @param array $tableData The structured array of event data to display.
 * @param array $t The translation array for the current language.
 * @param bool $isArchivePage Indicates if this is for a yearly archive.
 * @return string The HTML for the table.
 */
function generateEventTable($tableData, $t, $isArchivePage = false) {
    $monthNames = $t['month_names'];
    $html = "\t<table class='months-table'>\n\t\t<tr>\n";
    $monthCounter = 0;
    foreach ($tableData as $yearMonthKey => $dates) {
        $monthCounter++;
        list($year, $monthNum) = explode('-', $yearMonthKey);

        $isInitiallyVisible = ($isArchivePage || $monthCounter <= 3);
        $initialDisplay = $isInitiallyVisible ? 'block' : 'none';
        $initialArrow = $isInitiallyVisible ? '&#9650;' : '&#9660;';

        $html .= "\t\t\t<td class='month-cell'>\n";
        $html .= "\t\t\t\t<strong class='month-header' data-toggle-month='month-{$year}-{$monthNum}'>" . ($monthNames[$monthNum] ?? '') . "&nbsp;<span class='toggle-icon'>{$initialArrow}</span></strong>\n";
        $html .= "\t\t\t\t<div id='month-{$year}-{$monthNum}' class='month-content' style='display: {$initialDisplay};'>";

        foreach ($dates as $date) {
            $dateDir = basename(dirname($date));
            $timeCode = basename($date);
            
            $dayOfMonth = (int)substr($dateDir, 6, 2);
            $formattedTime = substr_replace(substr_replace($timeCode, ':', 4, 0), ':', 2, 0);

            if ($t['lang_short'] === 'en') {
                $formattedDay = formatDayWithSuffix($dayOfMonth);
                $formattedTimeForDisplay = "{$formattedDay} {$formattedTime}";
            } else {
                // This format (15.) works for nb, de, cs, fi
                $formattedDay = $dayOfMonth . '.';
                $formattedTimeForDisplay = "{$formattedDay} {$formattedTime}";
            }

            // Use language-prefixed file to check for multi-station status, fallback to default.
            $lang_prefix = ($t['lang_short'] === 'nb') ? '' : $t['lang_short'] . '_';
            $isMultiStation = file_exists("{$date}/{$lang_prefix}map.jpg") || file_exists("{$date}/map.jpg");
            
            $eventType = $isMultiStation ? 'multi-station-normal' : 'single-station';
            $linkClass = $isMultiStation ? 'normal-altitude' : 'unknown-altitude';
            
            // *** CORRECTION: Initialize $showerType outside the if-block ***
            $showerType = 'none';

            if ($isMultiStation) {
                // Use language-prefixed tables.html if it exists
                $tablesFile = "{$date}/{$lang_prefix}tables.html";
                if (!file_exists($tablesFile)) {
                    $tablesFile = "{$date}/tables.html"; // Fallback to default
                }
                
                if (file_exists($tablesFile)) {
                    $tablesContent = file_get_contents($tablesFile);
                    
                    $endAltitude = null;
                    $entrySpeed = null;
                    $startAltitude = null;
                    
                    // Regex keys based on default (Norwegian) translations as a stable fallback.
                    // This is safer as we don't know if all languages have the translation.
                    
                    // Try to find Start Altitude (Starthøgde)
                    if (preg_match('/(Start height|Starthøgde|Anfangshöhe|Počáteční výška|Alkukorkeus):<\/td><td>\s*([0-9,]+)\s*km/i', $tablesContent, $matchesStart)) {
                        $startAltitude = (float)str_replace(',', '.', $matchesStart[2]);
                    }

                    // Try to find End Altitude (Slutthøgde)
                    if (preg_match('/(End height|Slutthøgde|Endhöhe|Konečná výška|Loppukorkeus):<\/td><td>\s*([0-9,]+)\s*km/i', $tablesContent, $matchesAlt)) {
                        $endAltitude = (float)str_replace(',', '.', $matchesAlt[2]);
                    }

                    // Try to find Entry Speed (Inngangshastighet)
                    if (preg_match('/(Entry speed|Inngangshastighet|Eintrittsgeschwindigkeit|Vstupní rychlost|Tulonopeus):<\/td><td>\s*([0-9,]+)\s*km\/s/i', $tablesContent, $matchesSpeed)) {
                        $entrySpeed = (float)str_replace(',', '.', $matchesSpeed[2]);
                    }

                    // Check new criteria for meteorite candidate
                    if ($startAltitude !== null && $endAltitude !== null && $entrySpeed !== null &&
		        $startAltitude > 40 && $startAltitude < 120 &&
		     	$endAltitude > 10 && $endAltitude < 40 &&
    			$entrySpeed >= 10 && $entrySpeed <= 30) {
                        
                         $linkClass = 'low-altitude';
                         $eventType = 'meteorite-candidate';
                    }
                    
                    // Shower check (simple string search)
                    // $showerType = 'none'; // This was already initialized
                    if (stripos($tablesContent, 'Perseid') !== false) $showerType = 'perseider'; // Handles Perseidene, Perseidy, Perseiden, Perseidit
                    elseif (stripos($tablesContent, 'taurid') !== false) { // Handles Tauridene, Tauridy, Tauriden, Tauridit
                        if (stripos($tablesContent, 'Sørlige') !== false || stripos($tablesContent, 'Southern') !== false || stripos($tablesContent, 'Südliche') !== false || stripos($tablesContent, 'Jižní') !== false || stripos($tablesContent, 'Eteläiset') !== false) {
                             $showerType = 'sorlige-taurider';
                        } elseif (stripos($tablesContent, 'Nordlige') !== false || stripos($tablesContent, 'Northern') !== false || stripos($tablesContent, 'Nördliche') !== false || stripos($tablesContent, 'Severní') !== false || stripos($tablesContent, 'Pohjoiset') !== false) {
                             $showerType = 'nordlige-taurider';
                        }
                    }
                    elseif (stripos($tablesContent, 'Leonid') !== false) $showerType = 'leonider'; // Leonidit, Leoniden, Leonidy, Leonidene
                    elseif (stripos($tablesContent, 'Geminid') !== false) $showerType = 'geminider'; // Geminidit, Geminiden, Geminidy, Geminidene
                }
            }

            $locationInfo = ''; // Initialize
            if ($isMultiStation) {
                // Multi-station: Use location.txt
                $locationFile = "{$date}/location.txt";
                if (file_exists($locationFile)) {
                    $locationInfo = htmlspecialchars(trim(file_get_contents($locationFile)));
                }
            } else {
                // Single-station: Find station name and camera numbers
                $stationName = '';
                $camNumbers = [];
                
                // Use unprocessed images as the source of truth for detections
                $detectionPaths = glob("{$date}/*/*/fireball_orig.jpg"); 
                if (empty($detectionPaths)) {
                     $detectionPaths = glob("{$date}/*/*/fireball.jpg"); // Fallback to processed
                }

                foreach ($detectionPaths as $imgPath) {
                    $relativePath = substr($imgPath, strlen($date) + 1); // Remove "{$date}/"
                    $parts = explode('/', $relativePath); // e.g., [StationName, CamDir, FileName]
                    
                    if (count($parts) >= 2) {
                        if (empty($stationName)) {
                            $stationName = $parts[0]; // Set station name on first find
                        }
                        // Extract number from cam directory, e.g., "cam3" -> "3"
                        $camNum = preg_replace('/[^0-9]/', '', $parts[1]); 
                        if (!empty($camNum) && !in_array($camNum, $camNumbers)) {
                            $camNumbers[] = $camNum;
                        }
                    }
                }
                
                if (!empty($stationName)) {
                    sort($camNumbers, SORT_NUMERIC); // Sort them nicely, e.g., [3, 7]
                    // Format: (StationName 3, 7)
                    $locationInfo = '(' . htmlspecialchars($stationName) . ' ' . implode(', ', $camNumbers) . ')';
                }
            }


	    // Find processed and unprocessed images from subdirectories
	    $processedImages = glob("{$date}/*/*/fireball.jpg");
	    $unprocessedImages = glob("{$date}/*/*/fireball_orig.jpg");
	    $mediaHTML = '';

	    // Processed images
	    if (!empty($processedImages)) {
    	       $mediaHTML .= "<div class='media-container processed-images' style='display: none;'>";
    	       // We pass '.webm'.
    	       // generateMediaItem takes ".../fireball.jpg", strips ".jpg" -> ".../fireball",
    	       // and adds ".webm" -> ".../fireball.webm".
    	       foreach ($processedImages as $imgPath) $mediaHTML .= generateMediaItem($imgPath, '.webm', 'Processed fireball image');
    	       $mediaHTML .= "</div>";
	    }

	    // Unprocessed images
	    if (!empty($unprocessedImages)) {
	        $mediaHTML .= "<div class='media-container unprocessed-images' style='display: none;'>";
    		// We pass '_orig.webm'.
    		// generateMediaItem takes ".../fireball_orig.jpg", strips "_orig.jpg" -> ".../fireball",
    		// and adds "_orig.webm" -> ".../fireball_orig.webm".
    		foreach ($unprocessedImages as $imgPath) $mediaHTML .= generateMediaItem($imgPath, '_orig.webm', 'Unprocessed fireball image');
    		$mediaHTML .= "</div>";
	    }

            $html .= "<div class='event-container' data-event-type='{$eventType}' data-shower='{$showerType}'>";
            $html .= "<a href='/meteor/{$date}/' class='observation-link {$linkClass}'>";
            $html .= "<span>{$formattedTimeForDisplay}</span>";
            if ($locationInfo) {
                $html .= "<small class='location-details'>{$locationInfo}</small>";
            }
            $html .= "</a>";
            $html .= $mediaHTML;
            if (trim($mediaHTML) !== '') {
                $html .= "<hr class='image-separator'>";
            }
            $html .= "</div>\n";
        }
        $html .= "</div><span class='minimized-count' style='display: " . ($isInitiallyVisible ? 'none' : 'block') . ";'></span></td>\n";
    }
    $html .= "\t\t</tr>\n\t</table>\n";
    return $html;
}

/**
 * Generates the HTML footer, including the main JavaScript for interactivity.
 * @param array $t The translation array for the current language.
 * @return string The HTML footer string.
 */
function generatePageFooter($t) {
    $hidden_singular = $t['hidden_singular'];
    $hidden_plural = $t['hidden_plural'];
    return <<<HTML
    <script>
    (function() { // Wrap everything in an IIFE to avoid global scope pollution
        if (window.filterScriptLoaded) return;
        window.filterScriptLoaded = true;

        // --- Helper Functions ---
        // (These are unchanged)

        function loadImagesInContainer(container) {
            if (!container) return;
            const imagesToLoad = container.querySelectorAll('img[data-src]');
            imagesToLoad.forEach(img => {
                const src = img.getAttribute('data-src');
                if (src) {
                    img.src = src;
                    img.removeAttribute('data-src');
                }
            });
        }

        function updateMinimizedCounts() {
            const selectedFilterInput = document.querySelector('input[name="eventFilter"]:checked');
            if (!selectedFilterInput) return; // Guard against running too early
            const selectedFilter = selectedFilterInput.value;
            
            document.querySelectorAll('.month-cell').forEach(cell => {
                const content = cell.querySelector('.month-content');
                const countSpan = cell.querySelector('.minimized-count');
                if (!content || !countSpan) return;

                let count = 0;
                const eventsInMonth = content.querySelectorAll('.event-container');
                eventsInMonth.forEach(container => {
                    const eventType = container.getAttribute('data-event-type');
                    const showerType = container.getAttribute('data-shower');
                    let isVisibleInFilter = false;
                    const isMultiStation = (eventType === 'multi-station-normal' || eventType === 'meteorite-candidate');

                    switch (selectedFilter) {
                        case 'all': isVisibleInFilter = true; break;
                        case 'multi-station': isVisibleInFilter = isMultiStation; break;
                        case 'candidates': isVisibleInFilter = (eventType === 'meteorite-candidate'); break;
                        case 'perseider': case 'sorlige-taurider': case 'nordlige-taurider': case 'leonider': case 'geminider':
                            isVisibleInFilter = isMultiStation && (showerType === selectedFilter);
                            break;
                    }
                    if (isVisibleInFilter) count++;
                });
                countSpan.textContent = count > 0 ? (count === 1 ? '1 $hidden_singular' : count + ' $hidden_plural') : '';
            });
        }

        function filterEvents() {
            const selectedFilterInput = document.querySelector('input[name="eventFilter"]:checked');
            if (!selectedFilterInput) return; // Guard against running too early
            const selectedFilter = selectedFilterInput.value;
            
            document.querySelectorAll('.event-container').forEach(function(container) {
                const eventType = container.getAttribute('data-event-type');
                const showerType = container.getAttribute('data-shower');
                let show = false;
                const isMultiStation = (eventType === 'multi-station-normal' || eventType === 'meteorite-candidate');

                switch (selectedFilter) {
                    case 'all': show = true; break;
                    case 'multi-station': show = isMultiStation; break;
                    case 'candidates': show = (eventType === 'meteorite-candidate'); break;
                    case 'perseider': case 'sorlige-taurider': case 'nordlige-taurider': case 'leonider': case 'geminider':
                        show = isMultiStation && (showerType === selectedFilter);
                        break;
                }
                container.style.display = show ? 'block' : 'none';
            });
            updateMinimizedCounts();
        }

        function updateDisplayChoice() {
            const selectedDisplayInput = document.querySelector('input[name="displayChoice"]:checked');
            if (!selectedDisplayInput) return; // Guard against running too early
            const selectedDisplay = selectedDisplayInput.value;

            const showImages = selectedDisplay !== 'none';
            document.querySelectorAll('.media-container').forEach(c => c.style.display = 'none');
            if (showImages) {
                document.body.classList.add('images-are-shown');
                const targetClass = { 'processed': '.processed-images', 'unprocessed': '.unprocessed-images' }[selectedDisplay];
                if(targetClass) document.querySelectorAll(targetClass).forEach(c => c.style.display = 'block');
            } else {
                document.body.classList.remove('images-are-shown');
            }
        }

        // --- Setup Listeners for Clicks, Changes, Hovers ---
        // This *only* sets up future interactions.
        document.addEventListener('DOMContentLoaded', function() {
            const filterRadios = document.querySelectorAll('input[name="eventFilter"]');
            const displayRadios = document.querySelectorAll('input[name="displayChoice"]');

            // Attach all interaction listeners
            filterRadios.forEach(radio => radio.addEventListener('change', filterEvents));
            displayRadios.forEach(radio => radio.addEventListener('change', updateDisplayChoice));
            
            document.querySelectorAll('.month-header').forEach(header => {
                header.addEventListener('click', function() {
                    const monthContent = this.nextElementSibling;
                    const icon = this.querySelector('.toggle-icon');
                    const countSpan = this.parentElement.querySelector('.minimized-count');
                    if (monthContent && icon && countSpan) {
                        const isHidden = monthContent.style.display === 'none';
                        monthContent.style.display = isHidden ? 'block' : 'none';
                        icon.innerHTML = isHidden ? '&#9650;' : '&#9660;';
                        countSpan.style.display = isHidden ? 'none' : 'block';
                        if (isHidden) loadImagesInContainer(monthContent); // Load images when expanding
                        updateMinimizedCounts();
                    }
                });
            });
            
            document.body.addEventListener('mouseover', function(e) {
                const container = e.target.closest('.media-swap-container');
                if (container) {
                    const img = container.querySelector('img');
                    let video = container.querySelector('video');
                    if (!video && img && img.hasAttribute('data-videosrc')) {
                        video = document.createElement('video');
                        video.src = img.getAttribute('data-videosrc');
                        video.style.width = img.clientWidth + 'px';
                        video.style.height = img.clientHeight + 'px';
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
            
            // *** CRITICAL: The initial setup calls have been MOVED. ***
        });
        
        // --- Page Load & BFCache Restore Logic ---
        // 'pageshow' fires on *both* initial page load (event.persisted=false)
        // and on bfcache restore (event.persisted=true).
        window.addEventListener('pageshow', function(event) {
            
            // This is now the *only* place that runs the page setup logic.
            // It will read the radio buttons (either default or restored)
            // and apply the state, preventing any mismatch.
            
            // 1. Apply filters based on current (or restored) radio state
            filterEvents();
            
            // 2. Apply display choice based on current (or restored) radio state
            updateDisplayChoice();
            
            // 3. Load images for all *visible* month containers
            document.querySelectorAll('.month-content').forEach(content => {
                if (content.style.display === 'block') {
                     loadImagesInContainer(content);
                }
            });
        });

    })(); // End of IIFE
    </script>
HTML;
}

/**
 * Generates the complete HTML for a yearly archive page.
 * @param int $targetYear The year for the page title.
 * @param string $part The archive part ('a' or 'b').
 * @param array $t The translation array for the current language.
 * @return string The HTML header string.
 */
function generateArchivePageHeader($targetYear, $part, $t) {
    global $translations;

    $titleSuffix = ($part === 'a') ? $t['archive_part1_title'] : $t['archive_part2_title'];
    $pageTitle = sprintf($t['archive_title_for_year'], $targetYear) . ' ' . $titleSuffix;
    $mainTitle = htmlspecialchars($pageTitle);
    $backLinkText = '&laquo; ' . htmlspecialchars($t['archive_back_link_text']);
    $lang = htmlspecialchars($t['lang_short']);
    
    $langSwitcherHTML = '<div class="language-switcher">';
    // *** ADDED 'fi_FI' ***
    $flags = ['nb_NO' => '🇳🇴', 'en_GB' => '🇬🇧', 'de_DE' => '🇩🇪', 'cs_CZ' => '🇨🇿', 'fi_FI' => '🇫🇮'];
    foreach ($translations as $lang_code => $lang_t) {
        $lang_short = $lang_t['lang_short'];
        $archiveFileName = "{$targetYear}_{$lang_short}_{$part}.html";
        $langSwitcherHTML .= "<a href=\"./{$archiveFileName}\" title=\"{$lang_t['lang_name']}\">{$flags[$lang_code]}</a> ";
    }
    $langSwitcherHTML .= '</div>';

    return <<<HTML
<!DOCTYPE html>
<html lang="{$lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{$mainTitle}</title>
    <style>
        body { background: #ffffff; color: #000018; font-family: Verdana, Geneva, Arial, Helvetica, sans-serif; font-size: 12px; margin: 20px; }
        .page-header { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; margin-bottom: 10px; }
        .page-header h1 { grid-column: 1 / -1; grid-row: 1; text-align: center; margin: 0; color: #082060; }
        .language-switcher { grid-column: 3; grid-row: 1; justify-self: end; z-index: 1; }
        .language-switcher a { display: inline-block; text-decoration: none; margin: 0 2px; font-size: 1.5em; opacity: 0.7; transition: opacity 0.2s; }
        .language-switcher a:hover { opacity: 1; }

        @media screen and (max-width: 700px) {
            .page-header { grid-template-columns: 1fr; gap: 10px; }
            .page-header h1 { grid-row: 2; grid-column: 1; }
            .language-switcher { grid-row: 1; grid-column: 1; }
        }

        table { width: 100%; border-collapse: collapse; }
        td { vertical-align: top; }
        a { text-decoration: none; color: #c01010; }
        a:visited { color: #601010; }
        .filter-container { text-align: center; margin-bottom: 20px; }
        .filter-container fieldset { border: 1px solid #ccc; padding: 10px; margin: 10px; display: inline-block; vertical-align: top; }
        .filter-container legend { font-weight: bold; }
        .filter-container label { margin: 0 10px; display: block; text-align: left; }
        .months-table { border: 1px solid #ddd; }
        .month-cell { vertical-align: top; text-align: center; border: 1px solid #ddd; padding: 8px; }
        .month-header { font-weight: bold; border-bottom: 2px solid #000; margin-bottom: 5px; display: block; cursor: pointer; white-space: nowrap; }
        .month-header:hover { background-color: #f0f0f0; }
        .toggle-icon { font-size: 0.8em; margin-left: 5px; }
        .minimized-count { font-size: 0.8em; font-weight: normal; color: #666; display: block; font-style: italic; }
        .event-container { margin-bottom: 5px; padding-bottom: 5px; text-align: center; }
        .observation-link { display: block; }
        .observation-link span { display: block; }
        .location-details { font-size: 0.8em; }
        .media-container { margin-top: 5px; }
        .media-swap-container { display: inline-flex; flex-direction: column; align-items: center; gap: 5px; }
        .media-swap-container img, .media-swap-container video { margin: 2px; border: 1px solid #ccc; width: 256px; height: auto; vertical-align: top; }
        .image-separator { display: none; border: 0; border-top: 1px solid #888; margin: 15px 0 10px 0; }
        body.images-are-shown .image-separator { display: block; }
        .observation-link.low-altitude, .observation-link.low-altitude:visited { color: red; font-weight: bold; }
        .observation-link.normal-altitude, .observation-link.normal-altitude:visited { color: black; font-weight: bold; }
        .observation-link.unlikely-altitude, .observation-link.unlikely-altitude:visited { color: grey; }
        .observation-link.unknown-altitude, .observation-link.unknown-altitude:visited { color: black; font-weight: normal; }
        .back-link { text-align: center; display: block; margin-bottom: 10px; font-size: 1.2em; }
    </style>
</head>
<body>
    <div class="page-header">
        <h1>{$mainTitle}</h1>
        {$langSwitcherHTML}
    </div>
    <a href="/meteor/" class="back-link">{$backLinkText}</a>
HTML;
}

// --- Main Execution ---

if ($isYearArgProvided) {
    if (!preg_match('/^\d{4}$/', $argv[1])) {
        if ($isCliMode) {
            file_put_contents('php://stderr', "Error: Invalid argument. Please provide a 4-digit year (e.g., 2024).\n");
        }
        exit(1);
    }
    $targetYear = (int)$argv[1];

    $allEventData = gatherEventData([$targetYear]);

    foreach($translations as $lang_code => $t) {
        $lang_short = $t['lang_short'];

        // Part A: January - June
        $tableDataA = [];
        for ($m = 1; $m <= 6; $m++) {
            $monthNum = str_pad($m, 2, '0', STR_PAD_LEFT);
            if (isset($allEventData[$targetYear][$monthNum])) {
                $tableDataA["{$targetYear}-{$monthNum}"] = $allEventData[$targetYear][$monthNum];
            }
        }
        if (!empty($tableDataA)) {
            krsort($tableDataA);
            ob_start();
            echo generateArchivePageHeader($targetYear, 'a', $t);
            echo generateFilterControls($t);
            echo "<div class='year-container'>\n";
            echo generateEventTable($tableDataA, $t, true);
            echo "</div>\n";
            echo generatePageFooter($t);
            echo "</body>\n</html>";
            $outputFile = BASE_PATH . "/{$targetYear}_{$lang_short}_a.html";
            file_put_contents($outputFile, ob_get_clean());
        }

        // Part B: July - December
        $tableDataB = [];
        for ($m = 7; $m <= 12; $m++) {
            $monthNum = str_pad($m, 2, '0', STR_PAD_LEFT);
            if (isset($allEventData[$targetYear][$monthNum])) {
                $tableDataB["{$targetYear}-{$monthNum}"] = $allEventData[$targetYear][$monthNum];
            }
        }
        if (!empty($tableDataB)) {
            krsort($tableDataB);
            ob_start();
            echo generateArchivePageHeader($targetYear, 'b', $t);
            echo generateFilterControls($t);
            echo "<div class='year-container'>\n";
            echo generateEventTable($tableDataB, $t, true);
            echo "</div>\n";
            echo generatePageFooter($t);
            echo "</body>\n</html>";
            $outputFile = BASE_PATH . "/{$targetYear}_{$lang_short}_b.html";
            file_put_contents($outputFile, ob_get_clean());
        }
    }
} else {
    // CRON MAIN PAGE MODE (All languages)
    $currentYear = date('Y');
    $previousYear = $currentYear - 1;
    $allEventData = gatherEventData([$currentYear, $previousYear]);

    foreach($translations as $lang_code => $t) {
        $tableData = [];
        for ($i = 0; $i < 6; $i++) {
            $time = strtotime("-$i months");
            $year = date('Y', $time);
            $month = date('m', $time);
            if (isset($allEventData[$year][$month])) {
                $tableData["{$year}-{$month}"] = $allEventData[$year][$month];
            }
        }

        ob_start();
        if (!empty($tableData)) {
            echo generateFilterControls($t);
            echo "<div class='year-container'>\n\t<div class='year-header'>{$t['last_6_months']}</div>\n";
            echo generateEventTable($tableData, $t);
            echo "</div>\n";
            echo generatePageFooter($t);
        }
        $lang_short = $t['lang_short'];
        $outputFile = BASE_PATH . "/index-static_{$lang_short}.html";
        file_put_contents($outputFile, ob_get_clean());
    }
}
