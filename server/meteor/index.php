<?php
// --- Configuration ---
$DEFAULT_LANG = 'nb_NO';

// --- Translations for the main page shell ---
$translations = [
    'nb_NO' => [
        'page_title' => 'Meteorobservasjoner',
        'main_header' => 'De siste registreringene',
        'intro_p1' => 'Denne sida viser fortløpende registreringer fra kameranettverket. Kameranettverket er hovedsakelig finansiert av <a href="http://sparebankstiftelsendnb.no">Sparebankstiftelsen DNB</a>.',
        'about_table_header' => 'Om tabellen:',
        'about_table_li1' => '<strong>Grå markering:</strong> Hendelser registrert av kun én stasjon. Presis peiling er ikke mulig, og det kan være feilregistreringer (fly, fugl, etc.).',
        'about_table_li2' => '<strong>Rød markering:</strong> Meteorer observert lavere enn 40 km, som <i>kan</i> ha blitt meteoritter.',
        'about_table_li3' => 'Det kan ta noe tid fra en hendelse til den er ferdig prosessert, inntil noen timer eller enda lenger dersom manuell prosessering er nødvendig.',
        'about_table_li4' => 'Alle tider er i <strong>UTC</strong>. Legg til én time for norsk normaltid og to timer for sommertid. Klikk på tidspunktet for en detaljert rapport.',
        'your_observations_header' => 'Dine observasjoner:',
        'your_observations_p1' => 'Hvis du har sett en kraftig meteor som er registrert her, kan du rapportere den via vårt <a href="/obs.php">skjema</a>. Nye observasjoner kan gi mer presise resultater.',
        'general_info_header' => 'Generelt om meteorer:',
        'general_info_p1' => 'De fleste meteorer løser seg opp høgt i atmosfæren. Kun få (ca. 10 ganger i året i Norge) når bakken som meteoritter. De fleste av disse blir ikke observert på grunn av dagslys eller skyer.',
        'important_notice_p1' => '<strong>Viktig:</strong> Rapportene er automatisk genererte og kan ha varierende nøyaktighet. De viser ikke eventuelle nedfallsområder, da dette krever manuell kontroll og meteorologiske data.',
        'camera_image_alt' => 'Illustrasjon av meteorkamera',
        'archive_header' => 'Arkiv',
        'archive_for_year' => 'Registreringer for %d',
        'archive_part1' => '(januar til juni)',
        'archive_part2' => '(juli til desember)',
    ],
    'en_GB' => [
        'page_title' => 'Meteor Observations',
        'main_header' => 'Latest Detections',
        'intro_p1' => 'This page shows real-time detections from the camera network. The network is primarily funded by <a href="http://sparebankstiftelsendnb.no">Sparebankstiftelsen DNB</a>.',
        'about_table_header' => 'About the table:',
        'about_table_li1' => '<strong>Grey marking:</strong> Events recorded by only one station. A precise trajectory cannot be determined, and these may be false positives (planes, birds, etc.).',
        'about_table_li2' => '<strong>Red marking:</strong> Meteors terminating below 40 km altitude, which <i>may</i> have resulted in meteorites.',
        'about_table_li3' => 'It may take some time for an event to be fully processed, up to several hours or even longer if manual processing is required.',
        'about_table_li4' => 'All times are in <strong>UTC</strong>. Add one hour for standard local time (CET) and two hours for summer time (CEST). Click on a timestamp for a detailed report.',
        'your_observations_header' => 'Your observations:',
        'your_observations_p1' => 'If you have seen a bright meteor that is listed here, you can report it using our <a href="/obs.php">form</a>. New observations can lead to more accurate results.',
        'general_info_header' => 'General info about meteors:',
        'general_info_p1' => 'Most meteors burn up high in the atmosphere. Only a few (approx. 10 times a year in Norway) reach the ground as meteorites. Most of these go unobserved due to daylight or clouds.',
        'important_notice_p1' => '<strong>Important:</strong> The reports are automatically generated and may have varying degrees of accuracy. They do not show potential meteorite fall areas, as this requires manual verification and meteorological data.',
        'camera_image_alt' => 'Illustration of a meteor camera',
        'archive_header' => 'Archive',
        'archive_for_year' => 'Detections for %d',
        'archive_part1' => '(January to June)',
        'archive_part2' => '(July to December)',
    ],
    'de_DE' => [
        'page_title' => 'Meteorbeobachtungen',
        'main_header' => 'Neueste Erfassungen',
        'intro_p1' => 'Diese Seite zeigt Echtzeit-Erfassungen des Kameranetzwerks. Das Netzwerk wird hauptsächlich von der <a href="http://sparebankstiftelsendnb.no">Sparebankstiftelsen DNB</a> finanziert.',
        'about_table_header' => 'Über die Tabelle:',
        'about_table_li1' => '<strong>Graue Zeilen:</strong> Ereignisse, die nur von einer Station erfasst wurden. Eine genaue Flugbahn kann nicht bestimmt werden, und es kann sich um Fehlalarme handeln (Flugzeuge, Vögel usw.).',
        'about_table_li2' => '<strong>Rote Zeilen:</strong> Meteore, die unter 40 km Höhe beobachtet wurden und möglicherweise als Meteoriten niedergegangen sind.',
        'about_table_li3' => 'Die vollständige Verarbeitung eines Ereignisses kann einige Zeit in Anspruch nehmen, bis zu mehreren Stunden oder länger, wenn eine manuelle Bearbeitung erforderlich ist.',
        'about_table_li4' => 'Alle Zeiten sind in <strong>UTC</strong>. Fügen Sie eine Stunde für die lokale Standardzeit (MEZ) und zwei Stunden für die Sommerzeit (MESZ) hinzu. Klicken Sie auf einen Zeitstempel für einen detaillierten Bericht.',
        'your_observations_header' => 'Ihre Beobachtungen:',
        'your_observations_p1' => 'Wenn Sie einen hellen Meteor gesehen haben, der hier aufgeführt ist, können Sie ihn über unser <a href="/obs.php">Formular</a> melden. Neue Beobachtungen können zu genaueren Ergebnissen führen.',
        'general_info_header' => 'Allgemeine Informationen über Meteore:',
        'general_info_p1' => 'Die meisten Meteore verglühen hoch in der Atmosphäre. Nur wenige (ca. 10 Mal pro Jahr in Norwegen) erreichen den Boden als Meteoriten. Die meisten davon bleiben aufgrund von Tageslicht oder Wolken unbemerkt.',
        'important_notice_p1' => '<strong>Wichtig:</strong> Die Berichte werden automatisch erstellt und können unterschiedlich genaue sein. Sie zeigen keine potenziellen Einschlagsgebiete von Meteoriten, da dies eine manuelle Überprüfung und meteorologische Daten erfordert.',
        'camera_image_alt' => 'Illustration einer Meteorkamera',
        'archive_header' => 'Archiv',
        'archive_for_year' => 'Erfassungen für %d',
        'archive_part1' => '(Januar bis Juni)',
        'archive_part2' => '(Juli bis Dezember)',
    ],
    'cs_CZ' => [
        'page_title' => 'Pozorování meteorů',
        'main_header' => 'Nejnovější záznamy',
        'intro_p1' => 'Tato stránka zobrazuje záznamy z kamerové sítě v reálném čase. Síť je primárně financována <a href="http://sparebankstiftelsendnb.no">Sparebankstiftelsen DNB</a>.',
        'about_table_header' => 'O tabulce:',
        'about_table_li1' => '<strong>Šedé řádky:</strong> Události zaznamenané pouze jednou stanicí. Přesnou dráhu nelze určit a může se jednat o falešné záznamy (letadla, ptáci atd.).',
        'about_table_li2' => '<strong>Červené řádky:</strong> Bolidy pozorované v atmosféře níže než 40 km, které <i>mohly</i> dopadnout jako meteority.',
        'about_table_li3' => 'Zpracování události může nějakou dobu trvat, až několik hodin nebo i déle, pokud je nutné manuální zpracování.',
        'about_table_li4' => 'Všechny časy jsou v <strong>UTC</strong>. Pro místní standardní čas (SEČ) přičtěte jednu hodinu, pro letní čas (SELČ) dvě hodiny. Pro podrobnou zprávu klikněte na časový údaj.',
        'your_observations_header' => 'Vaše pozorování:',
        'your_observations_p1' => 'Pokud jste viděli jasný meteor, který je zde uveden, můžete ho nahlásit pomocí našeho <a href="/obs.php">formuláře</a>. Nová pozorování mohou vést k přesnějším výsledkům.',
        'general_info_header' => 'Obecné informace o meteorech:',
        'general_info_p1' => 'Většina meteorů shoří vysoko v atmosféře. Jen několik z nich (v Norsku přibližně 10krát za rok) dopadne na zem jako meteority. Většina z nich zůstane nespozorována kvůli dennímu světlu nebo oblačnosti.',
        'important_notice_p1' => '<strong>Důležité:</strong> Zprávy jsou generovány automaticky a jejich přesnost se může lišit. Neukazují potenciální oblasti dopadu meteoritů, protože to vyžaduje manuální ověření a meteorologická data.',
        'camera_image_alt' => 'Ilustrace meteorické kamery',
        'archive_header' => 'Archiv',
        'archive_for_year' => 'Záznamy za rok %d',
        'archive_part1' => '(leden - červen)',
        'archive_part2' => '(červenec - prosinec)',
    ],
    'fi_FI' => [
        'page_title' => 'Meteorihavainnot',
        'main_header' => 'Viimeisimmät havainnot',
        'intro_p1' => 'Tämä sivu näyttää kamerajärjestelmän reaaliaikaisia havaintoja. Järjestelmän päärahoittaja on <a href="http://sparebankstiftelsendnb.no">Sparebankstiftelsen DNB</a>.',
        'about_table_header' => 'Tietoja taulukosta:',
        'about_table_li1' => '<strong>Harmaa merkintä:</strong> Vain yhden aseman tallentamat tapahtumat. Tarkkaa lentorataa ei voida määrittää, ja kyseessä voi olla virheellinen havainto (lentokone, lintu tms.).',
        'about_table_li2' => '<strong>Punainen merkintä:</strong> Meteorit, joiden lento on päättynyt alle 40 km korkeuteen ja jotka <i>ovat saattaneet</i> pudota meteoriitteina.',
        'about_table_li3' => 'Tapahtuman täydellinen käsittely voi viedä aikaa, jopa useita tunteja tai pidempäänkin, jos manuaalinen käsittely on tarpeen.',
        // *** KORRIGERT LINJE FOR FINSKE TIDSZONER ***
        'about_table_li4' => 'Kaikki ajat ovat <strong>UTC</strong>-aikaa. Lisää kaksi tuntia Suomen normaaliaikaan (EET) ja kolme tuntia kesäaikaan (EEST). Klikkaamalla aikaleimaa näet yksityiskohtaisen raportin.',
        'your_observations_header' => 'Omat havaintosi:',
        'your_observations_p1' => 'Jos olet nähnyt kirkkaan meteorin, joka näkyy tässä luettelossa, voit raportoida sen <a href="/obs.php">lomakkeellamme</a>. Uudet havainnot voivat auttaa tarkentamaan tuloksia.',
        'general_info_header' => 'Yleistietoa meteoreista:',
        'general_info_p1' => 'Useimmat meteorit palavat loppuun korkealla ilmakehässä. Vain harvat (Norjassa noin 10 kertaa vuodessa) saavuttavat maanpinnan meteoriitteina. Suurin osa näistä jää havaitsematta päivänvalon tai pilvisyyden vuoksi.',
        'important_notice_p1' => '<strong>Tärkeää:</strong> Raportit luodaan automaattisesti, ja niiden tarkkuus voi vaihdella. Ne eivät näytä mahdollisia putoamisalueita, sillä se vaatii manuaalista tarkistusta ja meteorologisia tietoja.',
        'camera_image_alt' => 'Meteorikameran kuvituskuva',
        'archive_header' => 'Arkisto',
        'archive_for_year' => 'Havainnot vuodelta %d',
        'archive_part1' => '(tammi-kesäkuu)',
        'archive_part2' => '(heinä-joulukuu)',
    ],
];

// --- Setup ---
if (session_status() === PHP_SESSION_NONE) {
    session_start();
}

/**
 * Gets the user's real IP address, safely handling proxies.
 * @return string The user's IP address.
 */
function get_user_ip() {
    if (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {
        return trim(explode(',', $_SERVER['HTTP_X_FORWARDED_FOR'])[0]);
    }
    if (!empty($_SERVER['HTTP_X_REAL_IP'])) {
        return trim($_SERVER['HTTP_X_REAL_IP']);
    }
    return $_SERVER['REMOTE_ADDR'] ?? 'unknown_ip';
}

/**
 * Determines the desired language based on priority.
 * Priority: URL Param > Cookie > Browser Header > GeoIP > Default.
 * @param string $default_lang The default language code to use as a fallback.
 * @return string The determined and validated language code (e.g., 'en_GB').
 */
function get_language($default_lang) {
    // *** ADDED 'fi_FI' ***
    $supported_langs = ['nb_NO', 'en_GB', 'de_DE', 'cs_CZ', 'fi_FI'];
    if (isset($_GET['lang']) && in_array($_GET['lang'], $supported_langs)) {
        return $_GET['lang'];
    }
    if (isset($_COOKIE['lang']) && in_array($_COOKIE['lang'], $supported_langs)) {
        return $_COOKIE['lang'];
    }
    if (isset($_SERVER['HTTP_ACCEPT_LANGUAGE'])) {
        preg_match_all('/([a-z]{1,8}(-[a-z]{1,8})?)\s*(;\s*q\s*=\s*(1|0\.[0-9]+))?/i', $_SERVER['HTTP_ACCEPT_LANGUAGE'], $matches);
        if (count($matches[1])) {
            $langs = array_combine($matches[1], $matches[4]);
            foreach ($langs as $lang => $val) { if ($val === '') $langs[$lang] = 1; }
            arsort($langs, SORT_NUMERIC);
            foreach (array_keys($langs) as $browser_lang) {
                $browser_lang_code = str_replace('-', '_', $browser_lang);
                if (in_array($browser_lang_code, $supported_langs)) return $browser_lang_code;
                $short_code = substr($browser_lang_code, 0, 2);
                foreach ($supported_langs as $supported) {
                    if (substr($supported, 0, 2) === $short_code) return $supported;
                }
            }
        }
    }
    $country_to_lang_map = [
        'NO' => 'nb_NO', 'SE' => 'nb_NO', 'DK' => 'nb_NO',
        'GB' => 'en_GB', 'US' => 'en_GB', 'CA' => 'en_GB', 'AU' => 'en_GB', 'NZ' => 'en_GB', 'IE' => 'en_GB',
        'DE' => 'de_DE', 'AT' => 'de_DE', 'CH' => 'de_DE',
        'CZ' => 'cs_CZ', 'SK' => 'cs_CZ',
        'FI' => 'fi_FI', // *** ADDED 'fi_FI' ***
    ];
    $user_ip = get_user_ip();
    $geo_data_json = @file_get_contents("http://ip-api.com/json/{$user_ip}?fields=countryCode,status");
    if ($geo_data_json) {
        $geo_data = json_decode($geo_data_json);
        if ($geo_data && $geo_data->status === 'success' && isset($country_to_lang_map[$geo_data->countryCode])) {
            return $country_to_lang_map[$geo_data->countryCode];
        }
    }
    return $default_lang;
}

// --- Main Logic ---
$lang_code = get_language($DEFAULT_LANG);
setcookie('lang', $lang_code, time() + (86400 * 365), "/"); // Set cookie for 1 year

// Get the translation array for the selected language, falling back to default
$t = $translations[$lang_code] ?? $translations[$DEFAULT_LANG];

// Get the short 2-letter language code for the HTML lang attribute and for including the static file
$lang_short = substr($lang_code, 0, 2);

?>
<!DOCTYPE html>
<html lang="<?php echo htmlspecialchars($lang_short); ?>">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?php echo htmlspecialchars($t['page_title']); ?></title>
    <style>
        /* --- ORIGINAL STYLES --- */
        body { background: #ffffff; color: #000018; font-family: Verdana, Geneva, Arial, Helvetica, sans-serif; font-size: 12px; }
        h1 { text-align: center; color: #082060; }
        h2 { text-align: center; }
        table { width: 100%; border-collapse: collapse; }
        .submit input { border: 1px solid #000; background-color: #ffffff; color: #c01010; text-decoration: underline; font-family: Verdana, Geneva, Arial, Helvetica, sans-serif; font-size: 12px; overflow: visible; }
        td { vertical-align: top; }
        p { text-align: justify; }
        a { text-decoration: none; }
        hr { margin: 20px 0; }
        #observation-table-info { margin-bottom: 20px; }
        .spoiler { display: none; }

        /* --- GENERAL LINK STYLES --- */
        #observation-table-info a:link { color: #c01010; }
        #observation-table-info a:visited { color: #601010; }

        /* == STYLES FOR DYNAMICALLY GENERATED CONTENT == */
        .year-container { width: 100%; margin-bottom: 20px; }
        .year-header { text-align: center; font-size: 1.5em; font-weight: bold; padding-bottom: 10px; }
        .months-table { width: 100%; border-collapse: collapse; border: 1px solid #ddd; }
        .month-cell { vertical-align: top; text-align: center; border: 1px solid #ddd; padding: 8px; }
        .month-header { font-weight: bold; border-bottom: 2px solid #000; margin-bottom: 5px; display: block; cursor: pointer; }
        .toggle-icon { font-size: 0.8em; margin-left: 5px; }
        .minimized-count { font-size: 0.8em; font-weight: normal; color: #666; display: block; font-style: italic; }
        .observation-link { display: block; margin-bottom: 5px; padding-bottom: 5px; border-bottom: 1px solid #eee; }
        .observation-link:last-child { border-bottom: none; }
        .observation-link span { display: block; }
        .location-details { font-size: 0.8em; }

        /* === ADDED FOR IMAGE DISPLAY === */
        .media-container { margin-top: 5px; }
        .media-swap-container { display: flex; align-items: center; justify-content: center; gap: 5px; }
        .media-swap-container img, .media-swap-container video { margin: 2px; border: 1px solid #ccc; width: 256px; height: auto; vertical-align: top; }
        .image-separator { display: none; border: 0; border-top: 1px solid #888; margin: 15px 0 10px 0; }
        body.images-are-shown .image-separator { display: block; }
        
	    /* --- ALTITUDE COLOR DEFINITIONS --- */
	    .observation-link.low-altitude, .observation-link.low-altitude:visited { color: red; font-weight: bold; }
        .observation-link.normal-altitude, .observation-link.normal-altitude:visited { color: black; font-weight: bold; }
	    .observation-link.unlikely-altitude, .observation-link.unlikely-altitude:visited { color: grey; }
	    .observation-link.unknown-altitude, .observation-link.unknown-altitude:visited { color: black; font-weight: normal; }
        
        /* === ADDED STYLES FOR LAYOUT CONTROL === */
        .info-container { display: flex; align-items: flex-start; gap: 20px; padding: 0 15px; }
        .info-text { flex: 1; min-width: 60%; }
        .info-image { flex-shrink: 0; text-align: center; }
        .info-image img { max-width: 100%; height: auto; }
        .static-content-wrapper { padding: 0 15px; }
        .scrollable-table-container { overflow-x: auto; -webkit-overflow-scrolling: touch; }
        .scrollable-table-container img { width: 256px; height: auto; }
        .image-checkbox { display: none; }
        
        /* === STYLES FOR ARCHIVE LIST === */
        .archive-list-container { padding: 0 15px; }
        .archive-list {
            column-count: 3; /* Default to 3 columns */
            column-gap: 25px;
            list-style-type: none;
            padding: 0;
            margin: 0;
        }
        .archive-list a {
            display: inline-block; /* Treat link as a block but let it flow */
            width: 100%;
            padding: 4px 0;
            color: #c01010;
            text-decoration: underline;
            break-inside: avoid-column; /* Prevents items from breaking across columns */
        }
        .archive-list a:visited { color: #601010; }

        /* === RESPONSIVE HEADER LAYOUT === */
        .page-header { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; margin: 10px 15px; }
        .page-header h1 { grid-column: 1 / -1; grid-row: 1; text-align: center; margin: 0; }
        .language-switcher { grid-column: 3; grid-row: 1; justify-self: end; z-index: 1; }
        .language-switcher a { display: inline-block; text-decoration: none; margin: 0 2px; font-size: 1.5em; opacity: 0.7; transition: opacity 0.2s; }
        .language-switcher a:hover { opacity: 1; }

        /* === MEDIA QUERIES FOR RESPONSIVENESS === */
        @media screen and (max-width: 768px) {
            .info-container { flex-direction: column; }
            .archive-list { column-count: 2; } /* 2 columns for tablets */
        }

        @media screen and (max-width: 700px) {
            .page-header { grid-template-columns: 1fr; gap: 10px; }
            .page-header h1 { grid-row: 2; grid-column: 1; }
            .language-switcher { grid-row: 1; grid-column: 1; justify-self: end; }
        }

        @media screen and (max-width: 480px) {
            .archive-list { column-count: 1; } /* 1 column for mobile */
        }
    </style>
</head>
<body>
    <div class="page-header">
        <h1><?php echo $t['main_header']; ?></h1>
        <div class="language-switcher">
            <a href="?lang=nb_NO" title="Norsk">🇳🇴</a>
            <a href="?lang=en_GB" title="English">🇬🇧</a>
            <a href="?lang=de_DE" title="Deutsch">🇩🇪</a>
            <a href="?lang=cs_CZ" title="Čeština">🇨🇿</a>
            <a href="?lang=fi_FI" title="Suomi">🇫🇮</a> <?php // *** ADDED FINNISH FLAG *** ?>
        </div>
    </div>

    <div id="observation-table-info" class="info-container">
        <div class="info-text">
            <p><?php echo $t['intro_p1']; ?></p>

            <h3><?php echo $t['about_table_header']; ?></h3>
            <ul>
                <li><?php echo $t['about_table_li1']; ?></li>
                <li><?pHP echo $t['about_table_li2']; ?></li>
                <li><?php echo $t['about_table_li3']; ?></li>
                <li><?php echo $t['about_table_li4']; ?></li>
            </ul>

            <h3><?php echo $t['your_observations_header']; ?></h3>
            <p><?php echo $t['your_observations_p1']; ?></p>

            <h3><?php echo $t['general_info_header']; ?></h3>
            <p><?php echo $t['general_info_p1']; ?></p>
            <p><?php echo $t['important_notice_p1']; ?></p>

        </div>
        <div class="info-image">
            <img src="meteor_cam.jpg" width="512" alt="<?php echo htmlspecialchars($t['camera_image_alt']); ?>">
        </div>
    </div>

    <div class="static-content-wrapper">
        <div class="scrollable-table-container">
            <?php
                $static_file_path = "index-static_{$lang_short}.html";
                $default_static_file_path = 'index-static_nb.html'; // Fallback to Norwegian

                if (file_exists($static_file_path)) {
                    include($static_file_path);
                } elseif (file_exists($default_static_file_path)) {
                    include($default_static_file_path);
                }
            ?>
        </div>
    </div>

<hr>
    <h2><?php echo $t['archive_header']; ?></h2>
    <div class="archive-list-container">
        <div class="archive-list">
            <?php
            $currentYear = date('Y');
            for ($year = $currentYear; $year >= 2015; $year--) {
                $year_text = sprintf($t['archive_for_year'], $year);
                
                // Check for July-December file, linking to the language-specific version
                $filePathB = "/meteor/{$year}_{$lang_short}_b.html";
                $serverPathB = $_SERVER['DOCUMENT_ROOT'] . $filePathB;
                if (file_exists($serverPathB)) {
                    $link_text = htmlspecialchars($year_text . ' ' . $t['archive_part2']);
                    echo '<a href="' . $filePathB . '">' . $link_text . '</a>' . "\n";
                }

                // Check for January-June file, linking to the language-specific version
                $filePathA = "/meteor/{$year}_{$lang_short}_a.html";
                $serverPathA = $_SERVER['DOCUMENT_ROOT'] . $filePathA;
                if (file_exists($serverPathA)) {
                    $link_text = htmlspecialchars($year_text . ' ' . $t['archive_part1']);
                    echo '<a href="' . $filePathA . '">' . $link_text . '</a>' . "\n";
                }
            }
            ?>
        </div>
    </div>

</body>
</html>
