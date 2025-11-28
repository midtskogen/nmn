<?php
// --- Configuration ---
$DEFAULT_LANG = 'nb_NO';
$LANG_DIR = '/home/httpd/norskmeteornettverk.no/bin/loc'; // Corrected to a full, unambiguous path as a best practice.
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
 * Determines the desired language based on the same priorities as index.php.
 * Priority: URL Param > Cookie > Browser Header > GeoIP > Default.
 * @param string $default_lang The default language code to use as a fallback.
 * @return string The determined and validated language code (e.g., 'en_GB').
 */
function get_language($default_lang) {
    // ADDED 'fi_FI' to the list of supported languages
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
        'FI' => 'fi_FI', // ADDED Finland to the country map
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
setcookie('lang', $lang_code, time() + (86400 * 365), "/");

// Use the short 2-letter language code to build the filename (e.g., 'nb', 'en', 'fi')
// This matches the actual filenames like 'nb.json', 'en.json', 'fi.json', etc.
$lang_short = substr($lang_code, 0, 2);
$default_lang_short = substr($DEFAULT_LANG, 0, 2);

$t = [];
$lang_file_path = $LANG_DIR . '/' . $lang_short . '.json';
if (!file_exists($lang_file_path)) {
    // Fallback to the default language's short code if the detected one doesn't exist
    $lang_file_path = $LANG_DIR . '/' . $default_lang_short . '.json';
}

if (file_exists($lang_file_path)) {
    $t_json = file_get_contents($lang_file_path);
    if ($t_json) {
        $t = json_decode($t_json, true);
    }
}

// --- File Path Setup ---
$a = array_reverse(explode('/', getcwd()));
$path = "/meteor/" . $a[1] . "/" . $a[0] . "/";
$date = substr_replace($a[1], '-', 4, 0);
$date = substr_replace($date, '-', 7, 0);
$time = substr_replace($a[0], ':', 2, 0);
$time = substr_replace($time, ':', 5, 0);
$time = preg_replace("/[a-z]/", "", $time);
// The file prefix logic was already correct, using the short language code
$file_prefix = ($lang_short === 'nb') ? '' : $lang_short . '_';

$map_jpg_path        = "{$file_prefix}map.jpg";
$map_html_path       = "{$file_prefix}map.html";
$height_jpg_path     = "{$file_prefix}height.jpg";
$orbit_jpg_path      = "{$file_prefix}orbit.jpg";
$orbit_html_path     = "{$file_prefix}orbit.html";
$tables_html_path    = "{$file_prefix}tables.html";
$stations_html_path  = "{$file_prefix}stations.html";
$posvstime_jpg_path  = "{$file_prefix}posvstime.jpg";
$spd_acc_jpg_path    = "{$file_prefix}spd_acc.jpg";
$wind_jpg_path       = "{$file_prefix}wind_profile.jpg"; // Path for wind profile plot
$image_path          = "{$file_prefix}image.jpg";

/**
 * Checks if a language-specific file exists, falling back to the default (non-prefixed) file.
 * @param string $specific_path The path with the language prefix (e.g., 'en_map.jpg').
 * @param string $default_path The path without the language prefix (e.g., 'map.jpg').
 * @return string|null The accessible file path, or null if neither exists.
 */
function get_accessible_path($specific_path, $default_path) {
    return file_exists($specific_path) ? $specific_path : (file_exists($default_path) ? $default_path : null);
}

$map_jpg_display     = get_accessible_path($map_jpg_path, "map.jpg");
$map_html_display    = get_accessible_path($map_html_path, "map.html");
$height_jpg_display  = get_accessible_path($height_jpg_path, "height.jpg");
$orbit_jpg_display   = get_accessible_path($orbit_jpg_path, "orbit.jpg");
$orbit_html_display  = get_accessible_path($orbit_html_path, "orbit.html");
$tables_html_display   = get_accessible_path($tables_html_path, "tables.html");
$stations_html_display = get_accessible_path($stations_html_path, "stations.html");
$posvstime_jpg_display = get_accessible_path($posvstime_jpg_path, "posvstime.jpg");
$spd_acc_jpg_display   = get_accessible_path($spd_acc_jpg_path, "spd_acc.jpg");
$wind_jpg_display    = get_accessible_path($wind_jpg_path, "wind_profile.jpg"); // Check for wind profile
?>
<!DOCTYPE html>
<html lang="<?php echo htmlspecialchars($lang_short); ?>">
<head>
  <meta charset="UTF-8">
  <title><?php echo htmlspecialchars($t['report_title'] ?? 'Meteor Report'); ?></title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
<?php
print '<meta property="og:url" content="' . $path . '">' . "\n";
print '<meta property="og:type" content="article">' . "\n";
print '<meta property="og:site_name" content="Norsk meteornettverk">' . "\n";
$og_image_display = get_accessible_path($image_path, "image.jpg");
if ($og_image_display) {
    print '<meta property="og:image" content="' . $path . $og_image_display . '">' . "\n";
}
?>

<style>
:root {
  --primary-color: #082060; --secondary-color: #c01010; --background-color: #f4f6f9;
  --text-color: #333; --card-bg-color: #ffffff; --border-color: #dee2e6;
  --header-font: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; --body-font: 'Verdana', sans-serif;
}
body { font-family: var(--body-font); font-size: 16px; margin: 0; padding: 0;
 background-color: var(--background-color); color: var(--text-color); line-height: 1.6; }
.page-wrapper { margin: 0 auto; padding: 1em; position: relative; }
h1 { color: var(--primary-color);
 font-family: var(--header-font); font-size: 2.5em; margin: 0.5em 0; font-weight: 300; letter-spacing: -1px; text-align: center; }
p { text-align: justify;
}
a { color: var(--secondary-color); text-decoration: none; transition: color: 0.3s; }
a:hover { color: #601010; text-decoration: underline; }
img { max-width: 100%;
 height: auto; display: block; margin: 1em auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
.container { display: flex; flex-direction: row;
 gap: 2em; margin: 2em 0; align-items: flex-start; }
.column { flex: 1; display: flex; flex-direction: column; background-color: var(--card-bg-color); padding: 1.5em;
 border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
iframe { width: 100%; min-height: 512px; border: none; border-radius: 8px; flex-grow: 1;
}
@media (max-width: 1024px) { .container { flex-direction: column; } iframe { min-height: 512px; height: 512px; } }
.text-center { text-align: center;
}
table { width: 100%; border-collapse: collapse; margin-top: 1em; }
th, td { padding: 0.75em; text-align: left; border-bottom: 1px solid var(--border-color);
}
.container .column > table td:first-child { font-weight: bold; color: var(--primary-color); }
footer { text-align: center; margin-top: 2em; padding: 1em; color: #666;
 font-size: 0.9em; }

/* Language switcher positioned absolutely */
.language-switcher {
    position: absolute;
    top: 1em;
    right: 1em;
    z-index: 1000;
    background-color: transparent;
    padding: 0.25em 0.5em;
    border-radius: 8px;
    font-size: 1.5em;
}
.language-switcher a {
    text-decoration: none;
    margin: 0 0.25em;
    opacity: 0.7;
    transition: opacity 0.2s;
    display: inline-block;
    /* Ensures consistent behavior */
}
.language-switcher a:hover {
    opacity: 1;
}

/* ADDED: Restore auto-scaling for images inside tables from stations.html */
table img {
    width: 100% !important;
    /* Force override of inline width attribute */
    height: auto;
}
@media (max-width: 768px) {
  /* This targets the <td> elements of the main layout table 
    in stations.html. The [valign="top"] attribute is 
    a specific selector from that file's structure.
  */
  .column > table > tbody > tr > td[valign="top"] {
    display: block;  /* Makes the table cells stack vertically */
    width: 100%;     /* Ensures they take up the full width */
    box-sizing: border-box;
  }
}
</style>
</head>
<body>

<div class="page-wrapper">
    
    <div class="language-switcher">
        <span style="font-size: 0.7em;">
        <a href="?lang=nb_NO" title="Norsk">🇳🇴</a>
        <a href="?lang=en_GB" title="English">🇬🇧</a>
        <a href="?lang=de_DE" title="Deutsch">🇩🇪</a>
        <a href="?lang=cs_CZ" title="Čeština">🇨🇿</a>
        <a href="?lang=fi_FI" title="Suomi">🇫🇮</a> <?php // ADDED Finnish flag and link ?>
	</span>
    </div>

    <br>
    <h1>
    <?php
    if (file_exists('location.txt') && filesize('location.txt') > 0) {
        $location = trim(file_get_contents('location.txt'));
        $title_str = '';
        
        if ($lang_short === 'cs') {
            $declensions_path = $LANG_DIR . '/cs_declensions.json';
            $declined_location = null;
            if (file_exists($declensions_path)) {
                $declensions = json_decode(file_get_contents($declensions_path), true);
                if (isset($declensions[$location])) {
                    $declined_location = $declensions[$location];
                }
            }
            if ($declined_location) {
                $title_str = htmlspecialchars($t['meteor_over'] ?? 'Meteor nad') . ' ' . htmlspecialchars($declined_location);
            } else {
                $title_str = htmlspecialchars($t['meteor'] ?? 'Meteor') . ', ' . htmlspecialchars($t['location'] ?? 'poloha') . ': ' . htmlspecialchars($location);
            }
        } elseif ($lang_short === 'fi') {
            // HYBRID LOGIC: Check declension file first, then apply simple rule.
            $declensions_path = $LANG_DIR . '/fi_declensions.json';
            $declined_location = null;
            if (file_exists($declensions_path)) {
                $declensions = json_decode(file_get_contents($declensions_path), true);
                if (isset($declensions[$location])) {
                    $declined_location = $declensions[$location];
                }
            }

            // If not found in declensions, apply the simple fallback rule
            if ($declined_location === null) {
                $last_char = substr($location, -1);
                // Finnish vowels
                $vowels = ['a', 'e', 'i', 'o', 'u', 'y', 'ä', 'ö'];
                
                // Check against lowercase version of the last character
                if (in_array(mb_strtolower($last_char, 'UTF-8'), $vowels)) {
                    $declined_location = $location . 'n';
                } else {
                    $declined_location = $location . 'in';
                }
            }
            
            $title_str = htmlspecialchars($t['meteor_over'] ?? 'Meteor') . ' ' . htmlspecialchars($declined_location) . ' yllä';
            
        } else {
            // Default logic for nb, en, de
            $title_str = htmlspecialchars($t['meteor_over'] ?? 'Meteor over') . ' ' . htmlspecialchars($location);
        }
        
        echo $title_str . " <span style=\"white-space: nowrap;\">$date</span> $time " . htmlspecialchars($t['utc'] ?? 'UTC');
    } else {
        echo htmlspecialchars(($t['report_title'] ?? 'Meteor Report') . " $date $time " . $t['utc']);
    }
    ?>
    </h1>

    <?php // --- Container 1: Maps (Layout Change 2) --- ?>
    <?php if ($map_html_display || $map_jpg_display): // Show container if any map exists ?>
    <div class="container">
      
      <?php // Left Column: Primary Map (Interactive, or Static Fallback) ?>
      <div class="column">
        <?php if ($map_html_display): // Interactive Map exists ?>
            <div id="map-placeholder">
              <?php // The placeholder is empty, will be replaced by JS ?>
            </div>
            <script>document.addEventListener('DOMContentLoaded', function() { var ph = document.getElementById('map-placeholder'); if (ph) { var iframe = document.createElement('iframe'); iframe.src = '<?php echo $map_html_display; ?>'; ph.parentNode.replaceChild(iframe, ph); } });</script>
            <p class='text-center'>
                <a href='obs_<?php echo $date; ?>_<?php echo $time; ?>.kml'><?php echo htmlspecialchars($t['kml_file'] ?? 'KML file'); ?></a>
                &nbsp;|&nbsp;
                <a href='<?php echo $map_html_display; ?>'><?php echo htmlspecialchars($t['interactive_map'] ?? 'Interactive map'); ?></a>
            </p>
            
        <?php elseif ($map_jpg_display): // No Interactive map, but Static map exists ?>
            <p><?php echo htmlspecialchars($t['map_caption'] ?? 'Map caption missing.'); ?></p> <?php // Caption stays here in "Static Only" mode ?>
            <div id="map-placeholder">
                <a href='obs_<?php echo $date; ?>_<?php echo $time; ?>.kml'><img src='<?php echo $map_jpg_display; ?>' alt='map'></a>
            </div>
            <p class='text-center'><a href='obs_<?php echo $date; ?>_<?php echo $time; ?>.kml'><?php echo htmlspecialchars($t['kml_file'] ?? 'KML file'); ?></a></p>
            
        <?php endif; ?>
      </div>
      
      <?php // Right Column: Secondary Map (Static, if side-by-side) or Spacer ?>
      <?php if ($map_html_display && $map_jpg_display): // Side-by-side mode: Interactive (left) + Static (right) ?>
          <div class="column">
            <p><?php echo htmlspecialchars($t['map_caption'] ?? 'Map caption missing.'); ?></p> <?php // <-- CAPTION MOVED HERE ?>
            <img src='<?php echo $map_jpg_display; ?>' alt='static map'>
          </div>
          
      <?php elseif ($map_html_display || $map_jpg_display): // "Only" mode (Interactive Only OR Static Only) ?>
          <?php // We need this spacer to prevent the left column from taking 100% width ?>
          <div class="column" style="background: transparent; box-shadow: none; border: none; padding: 0;"></div>
      <?php endif; ?>
      
    </div>
    <?php endif; ?>


    <?php // --- Container 2: Height and Wind Profiles (Layout Changes 2 & 3) --- ?>
    <?php if ($height_jpg_display || $wind_jpg_display): // Show container if either exists ?>
    <div class="container">
      
      <?php // Left Column: Height plot (or empty spacer) ?>
      <?php if ($height_jpg_display): ?>
        <div class="column"><img src='<?php echo $height_jpg_display; ?>' alt='height profile'></div>
      <?php else: ?>
        <?php // Add an invisible spacer column to maintain the 50/50 layout ?>
        <div class="column" style="background: transparent; box-shadow: none; border: none; padding: 0;"></div>
      <?php endif; ?>
      
      <?php // Right Column: Wind plot (or empty spacer) ?>
      <?php if ($wind_jpg_display): ?>
        <div class="column"><img src='<?php echo $wind_jpg_display; ?>' alt='wind profile'></div>
      <?php else: ?>
        <?php // Add an invisible spacer column to maintain the 50/50 layout ?>
        <div class="column" style="background: transparent; box-shadow: none; border: none; padding: 0;"></div>
      <?php endif; ?>

    </div>
    <?php endif; ?>


    <?php if ($orbit_jpg_display || $orbit_html_display || $tables_html_display): ?>
    <div class="container">
      <?php if ($tables_html_display) { echo '<div class="column">'; include $tables_html_display; echo '</div>'; } ?>
      <?php if ($orbit_jpg_display || $orbit_html_display): ?>
      <div class="column">
        <div id="orbit-placeholder">
          <?php if ($orbit_jpg_display) echo "<img src='{$orbit_jpg_display}' alt='orbit plot'>"; ?>
        </div>
        <?php if ($orbit_html_display):
          echo "<script>document.addEventListener('DOMContentLoaded', function() { var ph = document.getElementById('orbit-placeholder'); if (ph) { var iframe = document.createElement('iframe'); iframe.src = '$orbit_html_display'; ph.parentNode.replaceChild(iframe, ph); } });</script>";
          echo "<p class='text-center'>";
          if ($orbit_jpg_display) {
             echo "<a href='{$orbit_jpg_display}'>" . htmlspecialchars($t['non_interactive_plot'] ?? 'Non-interactive plot') . "</a> | ";
          }
          echo "<a href='{$orbit_html_display}'>" . htmlspecialchars($t['interactive_plot'] ?? 'Interactive plot') . "</a>";
          echo "</p>";
        endif; ?>
      </div>
      <?php endif; ?>
    </div>
    <?php endif; ?>

    <?php if ($posvstime_jpg_display || $spd_acc_jpg_display): ?>
    <div class="container">
      <?php if ($posvstime_jpg_display): ?>
      <div class="column"><img src='<?php echo $posvstime_jpg_display; ?>' alt='position vs time'></div>
      <?php endif; ?>
      <?php if ($spd_acc_jpg_display): ?>
      <div class="column"><img src='<?php echo $spd_acc_jpg_display; ?>' alt='speed and acceleration'></div>
      <?php endif; ?>
    </div>
    <?php endif; ?>

    <hr style="border: none; border-top: 1px solid var(--border-color); margin: 2em 0;">

    <?php
    if ($stations_html_display) {
        // Alias $lang_short to $lang, as the included stations.html file
        // (generated by fetch.py) expects a variable named $lang for its internal PHP logic.
        $lang = $lang_short;
        include $stations_html_display;
    }
    ?>

    <footer>
        <p class="text-center"><?php echo htmlspecialchars($t['footer_text'] ?? 'This is an automatically generated report from the Norwegian Meteor Network.'); ?></p>
    </footer>

</div>
</body>
</html>
