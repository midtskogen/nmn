<?php

// --- Helper Functions ---

function sanitize_input($data) {
    if (is_array($data)) {
        return array_map('sanitize_input', $data);
    }
    return htmlspecialchars(stripslashes(trim($data)));
}

function generate_random_name($length = 5) {
    return substr(str_shuffle(str_repeat('0123456789abcdefghijklmnopqrstuvwxyz', ceil($length/36))),1,$length);
}

function save_base64_image($base64_string, $output_file) {
    $data = explode(',', $base64_string);
    if (count($data) < 2) return false;
    return file_put_contents($output_file, base64_decode($data[1])) !== false;
}

function normalize_coordinate_input($value) {
    if ($value === null) return '';
    return str_replace(',', '.', trim((string)$value));
}

function parse_and_validate_coordinate($value, $min, $max) {
    $normalized = normalize_coordinate_input($value);
    if ($normalized === '') return null;
    $parsed = filter_var($normalized, FILTER_VALIDATE_FLOAT);
    if ($parsed === false) return null;
    if ($parsed < $min || $parsed > $max) return null;
    return $parsed;
}

// --- Main Script Logic ---

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    
    // --- Translation Data ---
    $lang = sanitize_input($_POST['lang'] ?? 'nb_NO');
    $translations = [
        'nb_NO' => [
            'report_title' => 'Meteor-rapport',
            'thank_you_header' => 'Takk for rapporten!',
            'thank_you_subheader' => 'Her er ei oppsummering av observasjonen din.',
            'obs_time_and_place_header' => 'Observasjonstidspunkt og -sted',
            'time_utc_label' => 'Tidspunkt (UTC):',
            'accuracy_label' => 'Nøyaktighet:',
            'obs_location_label' => 'Observasjonssted:',
            'latitude_label' => 'Breddegrad',
            'longitude_label' => 'Lengdegrad',
            'meteor_path_header' => 'Meteorbane',
            'start_point_label' => 'Startpunkt:',
            'end_point_label' => 'Sluttpunkt:',
            'direction_label' => 'Retning',
            'altitude_label' => 'Høgde',
            'other_info_header' => 'Andre opplysninger',
            'color_label' => 'Dominerende farge:',
            'brightness_label' => 'Største lysstyrke:',
            'duration_label' => 'Varighet:',
            'other_phenomena_label' => 'Andre fenomen:',
            'no_phenomena_selected' => 'Ingen valgt.',
            'sound_delay_label' => 'Tid fra syn til lyd:',
            'observer_comments_label' => 'Observatørens kommentarer:',
            'contact_info_header' => 'Kontaktinformasjon',
            'name_label' => 'Navn:',
            'phone_label' => 'Telefon:',
            'email_label' => 'E-post:',
            'attachments_header' => 'Vedlegg',
            'report_id_label' => 'Rapport-ID',
            'uploaded_image_alt' => 'Opplastet bilde',
            'view_full_size' => 'Vis i full størrelse',
            'view_video' => 'Se video:',
            'email_subject' => 'Ny meteorrapport',
            'email_body_intro' => 'En ny meteorrapport har blitt sendt inn.',
            'email_full_report_link' => 'Full rapport',
            'email_comments_label' => 'Kommentarer',
            'time_accuracy' => ['unknown' => 'Veit ikke', 'pm1' => '±1 minutt', 'pm5' => '±5 minutt', 'pm15' => '±15 minutt', 'pm30' => '±30 minutt', 'gt30' => 'Mer enn 30 minutts usikkerhet'],
            'dominant_color' => ['unknown' => 'Usikker', 'white' => 'Hvit', 'green' => 'Grønn', 'blue' => 'Blå', 'yellow' => 'Gul', 'orange' => 'Oransje', 'red' => 'Rød', 'other' => 'Annen'],
            'brightness' => ['unknown' => 'Usikker', 'stars' => 'Som de klareste stjernene', 'brighter' => 'Litt sterkere enn stjernene', 'muchbrighter' => 'Mye sterkere enn stjernene men terrenget lyste ikke opp', 'lit' => 'Terrenget lyste opp', 'daylight' => 'Nesten som daglys', 'fulldaylight' => 'Fullt daglys'],
            'duration' => ['unknown' => 'Usikker', 'lt2' => 'under 2 sekund', '2-4' => '2 - 4 sekund', '4-8' => '4 - 8 sekund', '8-16' => '8 - 16 sekund', 'gt16' => 'mer enn 16 sekund'],
            'other_phenomena' => ['afterglow' => 'Etterglød', 'smoke' => 'Røykspor', 'fragmentation' => 'Oppsplitting', 'explosion' => 'Eksplosjon', 'sound' => 'Lyd/drønn'],
            'sound_delay' => ['unknown' => 'Usikker', 'lt30s' => 'Mindre enn 30 sekund', '30s-1m' => '30 sekund til ett minutt', '1m-1.5m' => 'Ett minutt til halvannet minutt', '1.5m-2m' => 'Halvannet minutt til to minutt', 'gt2m' => 'Mer enn to minutt'],
        ],
        'en_GB' => [
            'report_title' => 'Meteor Report',
            'thank_you_header' => 'Thank you for your report!',
            'thank_you_subheader' => 'Here is a summary of your observation.',
            'obs_time_and_place_header' => 'Time and Location of Observation',
            'time_utc_label' => 'Time (UTC):',
            'accuracy_label' => 'Accuracy:',
            'obs_location_label' => 'Observation Location:',
            'latitude_label' => 'Latitude',
            'longitude_label' => 'Longitude',
            'meteor_path_header' => 'Meteor Path',
            'start_point_label' => 'Start Point:',
            'end_point_label' => 'End Point:',
            'direction_label' => 'Direction',
            'altitude_label' => 'Altitude',
            'other_info_header' => 'Other Information',
            'color_label' => 'Dominant Color:',
            'brightness_label' => 'Peak Brightness:',
            'duration_label' => 'Duration:',
            'other_phenomena_label' => 'Other Phenomena:',
            'no_phenomena_selected' => 'None selected.',
            'sound_delay_label' => 'Time from Sight to Sound:',
            'observer_comments_label' => "Observer's Comments:",
            'contact_info_header' => 'Contact Information',
            'name_label' => 'Name:',
            'phone_label' => 'Phone:',
            'email_label' => 'E-mail:',
            'attachments_header' => 'Attachments',
            'report_id_label' => 'Report ID',
            'uploaded_image_alt' => 'Uploaded image',
            'view_full_size' => 'View full size',
            'view_video' => 'Watch video:',
            'email_subject' => 'New meteor report',
            'email_body_intro' => 'A new meteor report has been submitted.',
            'email_full_report_link' => 'Full report',
            'email_comments_label' => 'Comments',
            'time_accuracy' => ['unknown' => 'Unsure', 'pm1' => '±1 minute', 'pm5' => '±5 minutes', 'pm15' => '±15 minutes', 'pm30' => '±30 minutes', 'gt30' => 'More than 30 minutes uncertainty'],
            'dominant_color' => ['unknown' => 'Unsure', 'white' => 'White', 'green' => 'Green', 'blue' => 'Blue', 'yellow' => 'Yellow', 'orange' => 'Orange', 'red' => 'Red', 'other' => 'Other'],
            'brightness' => ['unknown' => 'Unsure', 'stars' => 'Like the brightest stars', 'brighter' => 'Slightly brighter than the stars', 'muchbrighter' => 'Much brighter than stars, but landscape not lit up', 'lit' => 'The landscape was lit up', 'daylight' => 'Almost like daylight', 'fulldaylight' => 'Full daylight'],
            'duration' => ['unknown' => 'Unsure', 'lt2' => 'less than 2 seconds', '2-4' => '2 - 4 seconds', '4-8' => '4 - 8 seconds', '8-16' => '8 - 16 seconds', 'gt16' => 'more than 16 seconds'],
            'other_phenomena' => ['afterglow' => 'Afterglow', 'smoke' => 'Smoke trail', 'fragmentation' => 'Fragmentation', 'explosion' => 'Explosion', 'sound' => 'Sound/boom'],
            'sound_delay' => ['unknown' => 'Unsure', 'lt30s' => 'Less than 30 seconds', '30s-1m' => '30 seconds to one minute', '1m-1.5m' => 'One to one and a half minutes', '1.5m-2m' => 'One and a half to two minutes', 'gt2m' => 'More than two minutes'],
        ],
        'de_DE' => [
            'report_title' => 'Meteor-Bericht',
            'thank_you_header' => 'Vielen Dank für Ihre Meldung!',
            'thank_you_subheader' => 'Hier ist eine Zusammenfassung Ihrer Beobachtung.',
            'obs_time_and_place_header' => 'Zeit und Ort der Beobachtung',
            'time_utc_label' => 'Zeit (UTC):',
            'accuracy_label' => 'Genauigkeit:',
            'obs_location_label' => 'Beobachtungsort:',
            'latitude_label' => 'Breitengrad',
            'longitude_label' => 'Längengrad',
            'meteor_path_header' => 'Meteorbahn',
            'start_point_label' => 'Startpunkt:',
            'end_point_label' => 'Endpunkt:',
            'direction_label' => 'Richtung',
            'altitude_label' => 'Höhe',
            'other_info_header' => 'Weitere Informationen',
            'color_label' => 'Dominante Farbe:',
            'brightness_label' => 'Maximale Helligkeit:',
            'duration_label' => 'Dauer:',
            'other_phenomena_label' => 'Weitere Phänomene:',
            'no_phenomena_selected' => 'Keine ausgewählt.',
            'sound_delay_label' => 'Zeit von Sichtung bis Geräusch:',
            'observer_comments_label' => 'Kommentare des Beobachters:',
            'contact_info_header' => 'Kontaktinformationen',
            'name_label' => 'Name:',
            'phone_label' => 'Telefon:',
            'email_label' => 'E-Mail:',
            'attachments_header' => 'Anhänge',
            'report_id_label' => 'Berichts-ID',
            'uploaded_image_alt' => 'Hochgeladenes Bild',
            'view_full_size' => 'Vollbild anzeigen',
            'view_video' => 'Video ansehen:',
            'email_subject' => 'Neuer Meteor-Bericht',
            'email_body_intro' => 'Ein neuer Meteor-Bericht wurde eingereicht.',
            'email_full_report_link' => 'Vollständiger Bericht',
            'email_comments_label' => 'Kommentare',
            'time_accuracy' => ['unknown' => 'Unsicher', 'pm1' => '±1 Minute', 'pm5' => '±5 Minuten', 'pm15' => '±15 Minuten', 'pm30' => '±30 Minuten', 'gt30' => 'Mehr als 30 Minuten Unsicherheit'],
            'dominant_color' => ['unknown' => 'Unsicher', 'white' => 'Weiß', 'green' => 'Grün', 'blue' => 'Blau', 'yellow' => 'Gelb', 'orange' => 'Orange', 'red' => 'Rot', 'other' => 'Andere'],
            'brightness' => ['unknown' => 'Unsicher', 'stars' => 'Wie die hellsten Sterne', 'brighter' => 'Etwas heller als die Sterne', 'muchbrighter' => 'Viel heller als Sterne, hat aber die Landschaft nicht erhellt', 'lit' => 'Die Landschaft wurde erhellt', 'daylight' => 'Fast wie Tageslicht', 'fulldaylight' => 'Volles Tageslicht'],
            'duration' => ['unknown' => 'Unsicher', 'lt2' => 'weniger als 2 Sekunden', '2-4' => '2 - 4 Sekunden', '4-8' => '4 - 8 Sekunden', '8-16' => '8 - 16 Sekunden', 'gt16' => 'mehr als 16 Sekunden'],
            'other_phenomena' => ['afterglow' => 'Nachglühen', 'smoke' => 'Rauchspur', 'fragmentation' => 'Fragmentierung', 'explosion' => 'Explosion', 'sound' => 'Geräusch/Knall'],
            'sound_delay' => ['unknown' => 'Unsicher', 'lt30s' => 'Weniger als 30 Sekunden', '30s-1m' => '30 Sekunden bis eine Minute', '1m-1.5m' => 'Eine bis anderthalb Minuten', '1.5m-2m' => 'Anderthalb bis zwei Minuten', 'gt2m' => 'Mehr als zwei Minuten'],
        ],
        'cs_CZ' => [
            'report_title' => 'Hlášení o meteoru',
            'thank_you_header' => 'Děkujeme za vaše hlášení!',
            'thank_you_subheader' => 'Zde je shrnutí vašeho pozorování.',
            'obs_time_and_place_header' => 'Čas a místo pozorování',
            'time_utc_label' => 'Čas (UTC):',
            'accuracy_label' => 'Přesnost:',
            'obs_location_label' => 'Místo pozorování:',
            'latitude_label' => 'Zeměpisná šířka',
            'longitude_label' => 'Zeměpisná délka',
            'meteor_path_header' => 'Dráha meteoru',
            'start_point_label' => 'Počáteční bod:',
            'end_point_label' => 'Koncový bod:',
            'direction_label' => 'Směr',
            'altitude_label' => 'Výška',
            'other_info_header' => 'Další informace',
            'color_label' => 'Dominantní barva:',
            'brightness_label' => 'Maximální jas:',
            'duration_label' => 'Doba trvání:',
            'other_phenomena_label' => 'Další jevy:',
            'no_phenomena_selected' => 'Žádné vybrány.',
            'sound_delay_label' => 'Doba od spatření po zvuk:',
            'observer_comments_label' => 'Komentáře pozorovatele:',
            'contact_info_header' => 'Kontaktní informace',
            'name_label' => 'Jméno:',
            'phone_label' => 'Telefon:',
            'email_label' => 'E-mail:',
            'attachments_header' => 'Přílohy',
            'report_id_label' => 'ID hlášení',
            'uploaded_image_alt' => 'Nahraný obrázek',
            'view_full_size' => 'Zobrazit v plné velikosti',
            'view_video' => 'Přehrát video:',
            'email_subject' => 'Nové hlášení o meteoru',
            'email_body_intro' => 'Bylo odesláno nové hlášení o meteoru.',
            'email_full_report_link' => 'Celé hlášení',
            'email_comments_label' => 'Komentáře',
            'time_accuracy' => ['unknown' => 'Nevím', 'pm1' => '±1 minuta', 'pm5' => '±5 minut', 'pm15' => '±15 minut', 'pm30' => '±30 minut', 'gt30' => 'Více než 30 minut nejistoty'],
            'dominant_color' => ['unknown' => 'Nevím', 'white' => 'Bílá', 'green' => 'Zelená', 'blue' => 'Modrá', 'yellow' => 'Žlutá', 'orange' => 'Oranžová', 'red' => 'Červená', 'other' => 'Jiná'],
            'brightness' => ['unknown' => 'Nevím', 'stars' => 'Jako nejjasnější hvězdy', 'brighter' => 'O něco jasnější než hvězdy', 'muchbrighter' => 'Mnohem jasnější než hvězdy, ale krajinu neosvětlil', 'lit' => 'Krajina byla osvětlena', 'daylight' => 'Téměř jako denní světlo', 'fulldaylight' => 'Plné denní světlo'],
            'duration' => ['unknown' => 'Nevím', 'lt2' => 'méně než 2 sekundy', '2-4' => '2 - 4 sekundy', '4-8' => '4 - 8 sekund', '8-16' => '8 - 16 sekund', 'gt16' => 'více než 16 sekund'],
            'other_phenomena' => ['afterglow' => 'Dosvit', 'smoke' => 'Kouřová stopa', 'fragmentation' => 'Rozpad', 'explosion' => 'Exploze', 'sound' => 'Zvuk/rána'],
            'sound_delay' => ['unknown' => 'Nevím', 'lt30s' => 'Méně než 30 sekund', '30s-1m' => '30 sekund až jedna minuta', '1m-1.5m' => 'Jedna až jedna a půl minuty', '1.5m-2m' => 'Jedna a půl až dvě minuty', 'gt2m' => 'Více než dvě minuty'],
        ],
        'fi_FI' => [
            'report_title' => 'Meteorihavaintoilmoitus',
            'thank_you_header' => 'Kiitos ilmoituksestasi!',
            'thank_you_subheader' => 'Tässä on yhteenveto havainnostasi.',
            'obs_time_and_place_header' => 'Havainnon aika ja paikka',
            'time_utc_label' => 'Aika (UTC):',
            'accuracy_label' => 'Tarkkuus:',
            'obs_location_label' => 'Havaintopaikka:',
            'latitude_label' => 'Leveyspiiri',
            'longitude_label' => 'Pituuspiiri',
            'meteor_path_header' => 'Meteorin lentorata',
            'start_point_label' => 'Alkupiste:',
            'end_point_label' => 'Loppupiste:',
            'direction_label' => 'Suunta',
            'altitude_label' => 'Korkeus',
            'other_info_header' => 'Muita tietoja',
            'color_label' => 'Hallitseva väri:',
            'brightness_label' => 'Suurin kirkkaus:',
            'duration_label' => 'Kesto:',
            'other_phenomena_label' => 'Muita ilmiöitä:',
            'no_phenomena_selected' => 'Ei valittu.',
            'sound_delay_label' => 'Aika havainnosta ääneen:',
            'observer_comments_label' => 'Havainnoitsijan kommentit:',
            'contact_info_header' => 'Yhteystiedot',
            'name_label' => 'Nimi:',
            'phone_label' => 'Puhelin:',
            'email_label' => 'Sähköposti:',
            'attachments_header' => 'Liitteet',
            'report_id_label' => 'Ilmoituksen ID',
            'uploaded_image_alt' => 'Ladattu kuva',
            'view_full_size' => 'Näytä täysikokoisena',
            'view_video' => 'Katso video:',
            'email_subject' => 'Uusi meteorihavaintoilmoitus',
            'email_body_intro' => 'Uusi meteorihavaintoilmoitus on lähetetty.',
            'email_full_report_link' => 'Koko ilmoitus',
            'email_comments_label' => 'Kommentit',
            'time_accuracy' => ['unknown' => 'Epävarma', 'pm1' => '±1 minuutti', 'pm5' => '±5 minuuttia', 'pm15' => '±15 minuuttia', 'pm30' => '±30 minuuttia', 'gt30' => 'Yli 30 minuutin epävarmuus'],
            'dominant_color' => ['unknown' => 'Epävarma', 'white' => 'Valkoinen', 'green' => 'Vihreä', 'blue' => 'Sininen', 'yellow' => 'Keltainen', 'orange' => 'Oranssi', 'red' => 'Punainen', 'other' => 'Muu'],
            'brightness' => ['unknown' => 'Epävarma', 'stars' => 'Kuin kirkkaimmat tähdet', 'brighter' => 'Hieman tähtiä kirkkaampi', 'muchbrighter' => 'Paljon tähtiä kirkkaampi, mutta ei valaissut maisemaa', 'lit' => 'Maisema valaistui', 'daylight' => 'Melkein kuin päivänvalo', 'fulldaylight' => 'Täysi päivänvalo'],
            'duration' => ['unknown' => 'Epävarma', 'lt2' => 'alle 2 sekuntia', '2-4' => '2–4 sekuntia', '4-8' => '4–8 sekuntia', '8-16' => '8–16 sekuntia', 'gt16' => 'yli 16 sekuntia'],
            'other_phenomena' => ['afterglow' => 'Jälkihohto', 'smoke' => 'Savuvana', 'fragmentation' => 'Hajoaminen', 'explosion' => 'Räjähdys', 'sound' => 'Ääni/pamaus'],
            'sound_delay' => ['unknown' => 'Epävarma', 'lt30s' => 'Alle 30 sekuntia', '30s-1m' => '30 sekunnista minuuttiin', '1m-1.5m' => 'Minuutista puoleentoista', '1.5m-2m' => 'Puolestatoista kahteen minuuttiin', 'gt2m' => 'Yli kaksi minuuttia'],
        ],
    ];

    // Translation helper function
    function t($key, $lang, $translations) {
        // Fallback to Norwegian if key or language not found
        return $translations[$lang][$key] ?? $translations['nb_NO'][$key] ?? $key;
    }

    // Translate a form value using the main translation array
    function translate_value($key, $value, $lang, $translations) {
        $values = $translations[$lang][$key] ?? $translations['nb_NO'][$key];
        return $values[$value] ?? htmlspecialchars($value);
    }

    // 1. Sanitize and retrieve form data
    $sighting_date_str = sanitize_input($_POST['sighting_date_full'] ?? '');
    $sighting_time_str = sanitize_input($_POST['sighting_time_full'] ?? '');
    $lat = parse_and_validate_coordinate($_POST['latitude'] ?? null, -90.0, 90.0);
    $lon = parse_and_validate_coordinate($_POST['longitude'] ?? null, -180.0, 180.0);
    if ($lat === null || $lon === null) {
        http_response_code(400);
        header('Content-Type: text/plain; charset=UTF-8');
        echo "Invalid coordinates.";
        exit();
    }
    $start_az = sanitize_input($_POST['bearing1'] ?? '');
    $start_alt = sanitize_input($_POST['alt1'] ?? '');
    $end_az = sanitize_input($_POST['bearing2'] ?? '');
    $end_alt = sanitize_input($_POST['alt2'] ?? '');
    $time_accuracy_val = sanitize_input($_POST['time_accuracy'] ?? 'unknown');
    $dominant_color_val = sanitize_input($_POST['dominant_color'] ?? 'unknown');
    $brightness_val = sanitize_input($_POST['brightness'] ?? 'unknown');
    $duration_val = sanitize_input($_POST['duration'] ?? 'unknown');
    
    // *** FIX: Ensure 'other_phenomena' is always treated as an array ***
    $other_phenomena = sanitize_input((array) ($_POST['other_phenomena'] ?? []));
    
    $sound_delay_val = sanitize_input($_POST['sound_delay'] ?? 'unknown');
    $more_info = sanitize_input($_POST['more_info'] ?? '');
    $contact_name = sanitize_input($_POST['contact_name'] ?? '');
    $contact_phone = sanitize_input($_POST['contact_phone'] ?? '');
    $contact_email = filter_var($_POST['contact_email'] ?? '', FILTER_SANITIZE_EMAIL);

    // 2. Create directory structure
    $date_obj = DateTime::createFromFormat('Y-m-d H:i:s', "{$sighting_date_str} {$sighting_time_str}") ?: new DateTime();
    $report_dir = 'reports/' . $date_obj->format('Ymd/His') . '/';
    if (!is_dir($report_dir)) mkdir($report_dir, 0777, true);

    // 3. Generate unique filename and paths
    $file_name = generate_random_name(5);
    $report_file_path = "{$report_dir}{$file_name}.html";
    
    // 4. Handle image uploads from base64 data
    $sky_view_image_name = '';
    if (isset($_POST['sky_view_image'])) {
        $sky_view_image_name = "sky-view-{$file_name}.png";
        save_base64_image($_POST['sky_view_image'], $report_dir . $sky_view_image_name);
    }
    
    $generated_map_image_name = '';
    if (isset($_POST['generated_map_image'])) {
        $generated_map_image_name = "map-view-{$file_name}.png";
        save_base64_image($_POST['generated_map_image'], $report_dir . $generated_map_image_name);
    }
    
    // 5. Handle File Uploads
    $uploaded_files_html = $uploaded_files_text = '';
    if (isset($_FILES['file_uploads'])) {
        $allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm'];
        foreach ($_FILES['file_uploads']['tmp_name'] as $i => $tmp_name) {
            if ($_FILES['file_uploads']['error'][$i] === UPLOAD_ERR_OK) {
                $file_type = mime_content_type($tmp_name);
                if (in_array($file_type, $allowed_types)) {
                    $original_name = $_FILES['file_uploads']['name'][$i];
                    $new_filename = "upload-{$i}-{$file_name}." . pathinfo($original_name, PATHINFO_EXTENSION);
                    if (move_uploaded_file($tmp_name, $report_dir . $new_filename)) {
                        $uploaded_files_text .= "- {$new_filename}\n";
                        if (strpos($file_type, 'image/') === 0) {
                            $uploaded_files_html .= "<div class='col-md-4 mb-3'><a href='{$new_filename}' target='_blank'><img src='{$new_filename}' class='img-fluid' alt='" . t('uploaded_image_alt', $lang, $translations) . "'></a><p class='text-center mt-1'><a href='{$new_filename}' target='_blank'>" . t('view_full_size', $lang, $translations) . "</a></p></div>";
                        } elseif (strpos($file_type, 'video/') === 0) {
                            $uploaded_files_html .= "<div class='col-md-4 mb-3'><p><a href='{$new_filename}' target='_blank'>" . t('view_video', $lang, $translations) . " {$new_filename}</a></p><video controls width='100%'><source src='{$new_filename}' type='{$file_type}'></video></div>";
                        }
                    }
                }
            }
        }
    }
    
    // 6. Prepare URLs and translated values for the template
    $protocol = 'https://'; // Force HTTPS to resolve mixed content issues
    $base_url = $protocol . $_SERVER['HTTP_HOST'];
    $absolute_report_url = "{$base_url}/{$report_file_path}";
    
    $time_accuracy_text = translate_value('time_accuracy', $time_accuracy_val, $lang, $translations);
    $dominant_color_text = translate_value('dominant_color', $dominant_color_val, $lang, $translations);
    $brightness_text = translate_value('brightness', $brightness_val, $lang, $translations);
    $duration_text = translate_value('duration', $duration_val, $lang, $translations);
    $sound_delay_text = translate_value('sound_delay', $sound_delay_val, $lang, $translations);
    
    $phenomena_list_html_items = '';
    foreach ($other_phenomena as $p) {
        $phenomena_list_html_items .= '<li>' . translate_value('other_phenomena', $p, $lang, $translations) . '</li>';
    }
    $phenomena_list_html = !empty($phenomena_list_html_items) ? '<ul>' . $phenomena_list_html_items . '</ul>' : '<p>' . t('no_phenomena_selected', $lang, $translations) . '</p>';
    
    $phenomena_list_text = !empty($other_phenomena) ? implode(', ', array_map(function($p) use ($lang, $translations) { return translate_value('other_phenomena', $p, $lang, $translations); }, $other_phenomena)) : t('no_phenomena_selected', $lang, $translations);
    
    $sound_delay_html = in_array('sound', $other_phenomena) ? "<h6>" . t('sound_delay_label', $lang, $translations) . "</h6><p>{$sound_delay_text}</p>" : '';
    $sound_delay_text_email = in_array('sound', $other_phenomena) ? t('sound_delay_label', $lang, $translations) . " {$sound_delay_text}\n" : '';
    
    // 7. Build the HTML content for the report file
    $html_template = <<<HTML
<!doctype html>
<html lang="{$lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{REPORT_TITLE}: {$date_obj->format('Y-m-d H:i')}</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"><style>body{background-color:#f8f9fa}.map-img,.sky-img,.card video,.card img{max-width:100%;width:100%;height:auto;border:1px solid #dee2e6;border-radius:.375rem}</style></head><body><div class="container-fluid my-4">{HEADER_PLACEHOLDER}<div class="row"><div class="col-lg-6 mb-4"><div class="card h-100"><div class="card-header"><h5>{OBS_TIME_PLACE_HEADER}</h5></div><div class="card-body"><h6>{TIME_UTC_LABEL}</h6><p>{$date_obj->format('Y-m-d H:i:s')} UTC</p><h6>{ACCURACY_LABEL}</h6><p>{$time_accuracy_text}</p><h6>{OBS_LOCATION_LABEL}</h6><p>{LATITUDE_LABEL}: {$lat}<br>{LONGITUDE_LABEL}: {$lon}</p><img src="{MAP_IMAGE_SRC}" alt="Map of observation location" class="img-fluid map-img mt-3"></div></div></div><div class="col-lg-6 mb-4"><div class="card h-100"><div class="card-header"><h5>{METEOR_PATH_HEADER}</h5></div><div class="card-body"><div class="row"><div class="col-sm-6"><h6>{START_POINT_LABEL}</h6><p>{DIRECTION_LABEL}: {$start_az}°<br>{ALTITUDE_LABEL}: {$start_alt}°</p></div><div class="col-sm-6"><h6>{END_POINT_LABEL}</h6><p>{DIRECTION_LABEL}: {$end_az}°<br>{ALTITUDE_LABEL}: {$end_alt}°</p></div></div><img src="{SKY_VIEW_IMAGE_URL}" alt="Sky map of observation" class="img-fluid sky-img mt-3"></div></div></div></div><div class="card mb-4"><div class="card-header"><h5>{OTHER_INFO_HEADER}</h5></div><div class="card-body"><h6>{COLOR_LABEL}</h6><p>{$dominant_color_text}</p><h6>{BRIGHTNESS_LABEL}</h6><p>{$brightness_text}</p><h6>{DURATION_LABEL}</h6><p>{$duration_text}</p><h6>{OTHER_PHENOMENA_LABEL}</h6>{$phenomena_list_html}{$sound_delay_html}<h6>{OBSERVER_COMMENTS_LABEL}</h6><p class="text-muted">{$more_info}</p></div></div><div class="card mb-4"><div class="card-header"><h5>{CONTACT_INFO_HEADER}</h5></div><div class="card-body"><p><strong>{NAME_LABEL}</strong> {$contact_name}</p><p><strong>{PHONE_LABEL}</strong> {$contact_phone}</p><p><strong>{EMAIL_LABEL}</strong> {$contact_email}</p></div></div><div class="card"><div class="card-header"><h5>{ATTACHMENTS_HEADER}</h5></div><div class="card-body"><div class="row">{$uploaded_files_html}</div></div></div><div class="text-center text-muted py-3">{REPORT_ID_LABEL}: {$file_name}</div></div></body></html>
HTML;
    
    // Replace all placeholders with translated text
    $placeholders = [
        '{REPORT_TITLE}' => t('report_title', $lang, $translations),
        '{OBS_TIME_PLACE_HEADER}' => t('obs_time_and_place_header', $lang, $translations),
        '{TIME_UTC_LABEL}' => t('time_utc_label', $lang, $translations),
        '{ACCURACY_LABEL}' => t('accuracy_label', $lang, $translations),
        '{OBS_LOCATION_LABEL}' => t('obs_location_label', $lang, $translations),
        '{LATITUDE_LABEL}' => t('latitude_label', $lang, $translations),
        '{LONGITUDE_LABEL}' => t('longitude_label', $lang, $translations),
        '{METEOR_PATH_HEADER}' => t('meteor_path_header', $lang, $translations),
        '{START_POINT_LABEL}' => t('start_point_label', $lang, $translations),
        '{END_POINT_LABEL}' => t('end_point_label', $lang, $translations),
        '{DIRECTION_LABEL}' => t('direction_label', $lang, $translations),
        '{ALTITUDE_LABEL}' => t('altitude_label', $lang, $translations),
        '{OTHER_INFO_HEADER}' => t('other_info_header', $lang, $translations),
        '{COLOR_LABEL}' => t('color_label', $lang, $translations),
        '{BRIGHTNESS_LABEL}' => t('brightness_label', $lang, $translations),
        '{DURATION_LABEL}' => t('duration_label', $lang, $translations),
        '{OTHER_PHENOMENA_LABEL}' => t('other_phenomena_label', $lang, $translations),
        '{OBSERVER_COMMENTS_LABEL}' => t('observer_comments_label', $lang, $translations),
        '{CONTACT_INFO_HEADER}' => t('contact_info_header', $lang, $translations),
        '{NAME_LABEL}' => t('name_label', $lang, $translations),
        '{PHONE_LABEL}' => t('phone_label', $lang, $translations),
        '{EMAIL_LABEL}' => t('email_label', $lang, $translations),
        '{ATTACHMENTS_HEADER}' => t('attachments_header', $lang, $translations),
        '{REPORT_ID_LABEL}' => t('report_id_label', $lang, $translations),
    ];
    $translated_html = str_replace(array_keys($placeholders), array_values($placeholders), $html_template);
    
    // 8. Create and save the final HTML report file
    $report_header = "<div class='text-center mb-4'><h1 class='display-4'>" . t('report_title', $lang, $translations) . "</h1></div>";
    $file_html = str_replace(['{HEADER_PLACEHOLDER}', '{MAP_IMAGE_SRC}', '{SKY_VIEW_IMAGE_URL}'], [$report_header, $generated_map_image_name, $sky_view_image_name], $translated_html);
    file_put_contents($report_file_path, $file_html);

    // 9. Send Email Notification (Forced to Norwegian)
    $to = "Steinar Midtskogen <steinar@norskmeteornettverk.no>, GEOTOP <mbgeotop@gmail.com>, Tor Einar Aslesen <taslesen@gmail.com>, Arne Danielsen <arne@soleskogobservatory.com>, Runar Sandnes <post@runarsandnes.com>, Vegard Lundby Rekaa <vegard@rekaa.no>";

    // Always use Norwegian for email subject and content
    $email_lang = 'nb_NO';
    $subject = t('email_subject', $email_lang, $translations) . " - " . $date_obj->format('Y-m-d H:i:s');

    // Re-translate values specifically for the Norwegian email
    $time_accuracy_text_no = translate_value('time_accuracy', $time_accuracy_val, $email_lang, $translations);
    $dominant_color_text_no = translate_value('dominant_color', $dominant_color_val, $email_lang, $translations);
    $brightness_text_no = translate_value('brightness', $brightness_val, $email_lang, $translations);
    $duration_text_no = translate_value('duration', $duration_val, $email_lang, $translations);
    $sound_delay_text_no = translate_value('sound_delay', $sound_delay_val, $email_lang, $translations);
    $phenomena_list_text_no = !empty($other_phenomena) ? implode(', ', array_map(function($p) use ($email_lang, $translations) { return translate_value('other_phenomena', $p, $email_lang, $translations); }, $other_phenomena)) : t('no_phenomena_selected', $email_lang, $translations);
    $sound_delay_text_email_no = in_array('sound', $other_phenomena) ? t('sound_delay_label', $email_lang, $translations) . " {$sound_delay_text_no}\n" : '';

    $message = t('email_body_intro', $email_lang, $translations) . "\n\n" .
        t('email_full_report_link', $email_lang, $translations) . ": {$absolute_report_url}\n\n" .
        t('time_utc_label', $email_lang, $translations) . " " . $date_obj->format('Y-m-d H:i:s') . "\n" .
        t('accuracy_label', $email_lang, $translations) . " " . $time_accuracy_text_no . "\n\n" .
        t('obs_location_label', $email_lang, $translations) . "\n" . t('latitude_label', $email_lang, $translations) . ": {$lat}\n" . t('longitude_label', $email_lang, $translations) . ": {$lon}\n\n" .
        t('meteor_path_header', $email_lang, $translations) . ":\n" . t('start_point_label', $email_lang, $translations) . " {$start_az}° Az, {$start_alt}° Alt\n" . t('end_point_label', $email_lang, $translations) . " {$end_az}° Az, {$end_alt}° Alt\n\n" .
        t('other_info_header', $email_lang, $translations) . ":\n" . t('color_label', $email_lang, $translations) . " " . $dominant_color_text_no . "\n" . t('brightness_label', $email_lang, $translations) . " " . $brightness_text_no . "\n" . t('duration_label', $email_lang, $translations) . " " . $duration_text_no . "\n" . t('other_phenomena_label', $email_lang, $translations) . " " . $phenomena_list_text_no . "\n{$sound_delay_text_email_no}" .
        t('email_comments_label', $email_lang, $translations) . ":\n{$more_info}\n\n" .
        t('contact_info_header', $email_lang, $translations) . ":\n" . t('name_label', $email_lang, $translations) . " {$contact_name}\n" . t('phone_label', $email_lang, $translations) . " {$contact_phone}\n" . t('email_label', $email_lang, $translations) . " {$contact_email}\n\n" .
        t('attachments_header', $email_lang, $translations) . ":\n{$uploaded_files_text}";

    // Set Reply-To address dynamically
    $reply_to_address = !empty($contact_email) && filter_var($contact_email, FILTER_VALIDATE_EMAIL) ? $contact_email : 'steinar@norskmeteornettverk.no';
    $headers = "From: Norsk meteornettverk <steinar@norskmeteornettverk.no>\r\nReply-To: {$reply_to_address}\r\nContent-Type: text/plain; charset=UTF-8\r\nX-Mailer: PHP/" . phpversion();
    mail($to, $subject, $message, $headers);

    // 10. Echo the confirmation HTML back to the user
    $confirmation_header_html = "<div class='text-center mb-4'><h1 class='display-4'>" . t('thank_you_header', $lang, $translations) . "</h1><p class='lead'>" . t('thank_you_subheader', $lang, $translations) . "</p></div>";
    $confirmation_html = str_replace(
        ['{HEADER_PLACEHOLDER}', '{MAP_IMAGE_SRC}', '{SKY_VIEW_IMAGE_URL}'],
        [$confirmation_header_html, "{$base_url}/{$report_dir}{$generated_map_image_name}", "{$base_url}/{$report_dir}{$sky_view_image_name}"],
        $translated_html
    );
    $confirmation_html = preg_replace("/(src|href)='(upload-)/", "$1='{$base_url}/{$report_dir}$2", $confirmation_html);
    echo $confirmation_html;

} else {
    // Redirect if accessed directly
    header("Location: obs.html");
    exit();
}
?>


