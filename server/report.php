<?php

// --- Helper Functions ---

function sanitize_input($data) {
    return htmlspecialchars(stripslashes(trim($data)));
}

function generate_random_name($length = 5) {
    return substr(str_shuffle(str_repeat('0123456789abcdefghijklmnopqrstuvwxyz', ceil($length/36))),1,$length);
}

function translate_value($key, $value) {
    $translations = [
        'time_accuracy' => ['unknown' => 'Veit ikke', 'pm1' => '±1 minutt', 'pm5' => '±5 minutt', 'pm15' => '±15 minutt', 'pm30' => '±30 minutt', 'gt30' => 'Mer enn 30 minutts usikkerhet'],
        'dominant_color' => ['unknown' => 'Usikker', 'white' => 'Hvit', 'green' => 'Grønn', 'blue' => 'Blå', 'yellow' => 'Gul', 'orange' => 'Oransje', 'red' => 'Rød', 'other' => 'Annen'],
        'brightness' => ['unknown' => 'Usikker', 'stars' => 'Som de klareste stjernene', 'brighter' => 'Litt sterkere enn stjernene', 'muchbrighter' => 'Mye sterkere enn stjernene men terrenget lyste ikke opp', 'lit' => 'Terrenget lyste opp', 'daylight' => 'Nesten som daglys', 'fulldaylight' => 'Fullt daglys'],
        'duration' => ['unknown' => 'Usikker', 'lt2' => 'under 2 sekund', '2-4' => '2 - 4 sekund', '4-8' => '4 - 8 sekund', '8-16' => '8 - 16 sekund', 'gt16' => 'mer enn 16 sekund'],
        'other_phenomena' => ['afterglow' => 'Etterglød', 'smoke' => 'Røykspor', 'fragmentation' => 'Oppsplitting', 'explosion' => 'Eksplosjon', 'sound' => 'Lyd/drønn'],
        'sound_delay' => ['unknown' => 'Usikker', 'lt30s' => 'Mindre enn 30 sekund', '30s-1m' => '30 sekund til ett minutt', '1m-1.5m' => 'Ett minutt til halvannet minutt', '1.5m-2m' => 'Halvannet minutt til to minutt', 'gt2m' => 'Mer enn to minutt'],
    ];
    return $translations[$key][$value] ?? htmlspecialchars($value);
}

function save_base64_image($base64_string, $output_file) {
    $data = explode(',', $base64_string);
    if (count($data) < 2) return false;
    return file_put_contents($output_file, base64_decode($data[1])) !== false;
}

// --- Main Script Logic ---

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // 1. Sanitize and retrieve form data
    $sighting_date_str = sanitize_input($_POST['sighting_date_full'] ?? '');
    $sighting_time_str = sanitize_input($_POST['sighting_time_full'] ?? '');
    $lat = filter_var($_POST['latitude'] ?? '0', FILTER_SANITIZE_NUMBER_FLOAT, FILTER_FLAG_ALLOW_FRACTION);
    $lon = filter_var($_POST['longitude'] ?? '0', FILTER_SANITIZE_NUMBER_FLOAT, FILTER_FLAG_ALLOW_FRACTION);
    $start_az = sanitize_input($_POST['bearing1'] ?? '');
    $start_alt = sanitize_input($_POST['alt1'] ?? '');
    $end_az = sanitize_input($_POST['bearing2'] ?? '');
    $end_alt = sanitize_input($_POST['alt2'] ?? '');
    $time_accuracy_val = sanitize_input($_POST['time_accuracy'] ?? 'unknown');
    $dominant_color_val = sanitize_input($_POST['dominant_color'] ?? 'unknown');
    $brightness_val = sanitize_input($_POST['brightness'] ?? 'unknown');
    $duration_val = sanitize_input($_POST['duration'] ?? 'unknown');
    $other_phenomena = isset($_POST['other_phenomena']) ? (array)$_POST['other_phenomena'] : [];
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
    
    // NEW: Handle the generated map image
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
                            $uploaded_files_html .= "<div class='col-md-4 mb-3'><a href='{$new_filename}' target='_blank'><img src='{$new_filename}' class='img-fluid' alt='Opplastet bilde'></a><p class='text-center mt-1'><a href='{$new_filename}' target='_blank'>Vis i full størrelse</a></p></div>";
                        } elseif (strpos($file_type, 'video/') === 0) {
                            $uploaded_files_html .= "<div class='col-md-4 mb-3'><p><a href='{$new_filename}' target='_blank'>Se video: {$new_filename}</a></p><video controls width='100%'><source src='{$new_filename}' type='{$file_type}'></video></div>";
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
    
    $time_accuracy_text = translate_value('time_accuracy', $time_accuracy_val);
    $dominant_color_text = translate_value('dominant_color', $dominant_color_val);
    $brightness_text = translate_value('brightness', $brightness_val);
    $duration_text = translate_value('duration', $duration_val);
    $sound_delay_text = translate_value('sound_delay', $sound_delay_val);
    $phenomena_list_html = !empty($other_phenomena) ? '<ul>' . implode('', array_map(function($p) { return '<li>' . translate_value('other_phenomena', sanitize_input($p)) . '</li>'; }, $other_phenomena)) . '</ul>' : '<p>Ingen valgt.</p>';
    $phenomena_list_text = !empty($other_phenomena) ? implode(', ', array_map(function($p) { return translate_value('other_phenomena', sanitize_input($p)); }, $other_phenomena)) : 'Ingen valgt.';
    $sound_delay_html = in_array('sound', $other_phenomena) ? "<h6>Tid fra syn til lyd:</h6><p>{$sound_delay_text}</p>" : '';
    $sound_delay_text_email = in_array('sound', $other_phenomena) ? "Lydforsinkelse: {$sound_delay_text}\n" : '';
    
    // 7. Build the HTML content for the report file
    $html_template = <<<HTML
<!doctype html>
<html lang="no"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Meteor-rapport: {$date_obj->format('Y-m-d H:i')}</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"><style>body{background-color:#f8f9fa}.map-img,.sky-img,.card video,.card img{max-width:100%;width:100%;height:auto;border:1px solid #dee2e6;border-radius:.375rem}</style></head><body><div class="container-fluid my-4">{HEADER_PLACEHOLDER}<div class="row"><div class="col-lg-6 mb-4"><div class="card h-100"><div class="card-header"><h5>Observasjonstidspunkt og -sted</h5></div><div class="card-body"><h6>Tidspunkt (UTC):</h6><p>{$date_obj->format('d. F Y, H:i:s')} UTC</p><h6>Nøyaktighet:</h6><p>{$time_accuracy_text}</p><h6>Observasjonssted:</h6><p>Breddegrad: {$lat}<br>Lengdegrad: {$lon}</p><img src="{MAP_IMAGE_SRC}" alt="Kart over observasjonssted" class="img-fluid map-img mt-3"></div></div></div><div class="col-lg-6 mb-4"><div class="card h-100"><div class="card-header"><h5>Meteorbane</h5></div><div class="card-body"><div class="row"><div class="col-sm-6"><h6>Startpunkt:</h6><p>Retning: {$start_az}°<br>Høgde: {$start_alt}°</p></div><div class="col-sm-6"><h6>Sluttpunkt:</h6><p>Retning: {$end_az}°<br>Høgde: {$end_alt}°</p></div></div><img src="{SKY_VIEW_IMAGE_URL}" alt="Himmelkart av observasjon" class="img-fluid sky-img mt-3"></div></div></div></div><div class="card mb-4"><div class="card-header"><h5>Andre opplysninger</h5></div><div class="card-body"><h6>Dominerende farge:</h6><p>{$dominant_color_text}</p><h6>Største lysstyrke:</h6><p>{$brightness_text}</p><h6>Varighet:</h6><p>{$duration_text}</p><h6>Andre fenomen:</h6>{$phenomena_list_html}{$sound_delay_html}<h6>Observatørens kommentarer:</h6><p class="text-muted">{$more_info}</p></div></div><div class="card mb-4"><div class="card-header"><h5>Kontaktinformasjon</h5></div><div class="card-body"><p><strong>Navn:</strong> {$contact_name}</p><p><strong>Telefon:</strong> {$contact_phone}</p><p><strong>E-post:</strong> {$contact_email}</p></div></div><div class="card"><div class="card-header"><h5>Vedlegg</h5></div><div class="card-body"><div class="row">{$uploaded_files_html}</div></div></div><div class="text-center text-muted py-3">Rapport-ID: {$file_name}</div></div></body></html>
HTML;
    
    // 8. Create and save the final HTML report file
    $file_html = str_replace(['{HEADER_PLACEHOLDER}', '{MAP_IMAGE_SRC}', '{SKY_VIEW_IMAGE_URL}'], ['<div class="text-center mb-4"><h1 class="display-4">Meteor-rapport</h1></div>', $generated_map_image_name, $sky_view_image_name], $html_template);
    file_put_contents($report_file_path, $file_html);

    // 9. Send Email Notification
    $to = "Steinar Midtskogen <steinar@norskmeteornettverk.no>, GEOTOP <mbgeotop@gmail.com>, Tor Einar Aslesen <taslesen@gmail.com>, Arne Danielsen <arne@soleskogobservatory.com>, Runar Sandnes <post@runarsandnes.com>, Vegard Lundby Rekaa <vegard@rekaa.no>";
    $subject = "Ny meteorrapport - " . $date_obj->format('Y-m-d H:i:s');
    $message = "En ny meteorrapport har blitt sendt inn.\n\nFull rapport: {$absolute_report_url}\n\nTidspunkt (UTC): {$date_obj->format('d. F Y, H:i:s')}\nNøyaktighet: {$time_accuracy_text}\n\nObservasjonssted:\nBreddegrad: {$lat}\nLengdegrad: {$lon}\n\nBane:\nStart: {$start_az}° Az, {$start_alt}° Alt\nSlutt: {$end_az}° Az, {$end_alt}° Alt\n\nAndre opplysninger:\nFarge: {$dominant_color_text}\nLysstyrke: {$brightness_text}\nVarighet: {$duration_text}\nFenomen: {$phenomena_list_text}\n{$sound_delay_text_email}Kommentarer:\n{$more_info}\n\nKontaktinformasjon:\nNavn: {$contact_name}\nTelefon: {$contact_phone}\nE-post: {$contact_email}\n\nVedlegg:\n{$uploaded_files_text}";
    $headers = "From: Norsk meteornettverk <steinar@norskmeteornettverk.no>\r\nReply-To: steinar@norskmeteornettverk.no\r\nContent-Type: text/plain; charset=UTF-8\r\nX-Mailer: PHP/" . phpversion();
    mail($to, $subject, $message, $headers);

    // 10. Echo the confirmation HTML back to the user
    $confirmation_html = str_replace(
        ['{HEADER_PLACEHOLDER}', '{MAP_IMAGE_SRC}', '{SKY_VIEW_IMAGE_URL}'],
        ['<div class="text-center mb-4"><h1 class="display-4">Takk for rapporten!</h1><p class="lead">Her er ei oppsummering av observasjonen din.</p></div>', "{$base_url}/{$report_dir}{$generated_map_image_name}", "{$base_url}/{$report_dir}{$sky_view_image_name}"],
        $html_template
    );
    $confirmation_html = preg_replace("/(src|href)='(upload-)/", "$1='{$base_url}/{$report_dir}$2", $confirmation_html);
    echo $confirmation_html;

} else {
    header("Location: obs4.html");
    exit();
}
?>
