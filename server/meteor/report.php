<?php
// Suggestion 4 — Combined design:
// - MULTI-STATION: station thumbnail strip (static images, gnomonic preferred) →
//   scientific data section → tabbed station detail (image left, links table right)
// - SINGLE STATION: falls back to the original index.php layout

$DEFAULT_LANG = 'nb_NO';
$LANG_DIR = '/home/httpd/norskmeteornettverk.no/bin/loc';
if (session_status() === PHP_SESSION_NONE) { session_start(); }

function s4_get_user_ip() {
    if (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) return trim(explode(',', $_SERVER['HTTP_X_FORWARDED_FOR'])[0]);
    if (!empty($_SERVER['HTTP_X_REAL_IP'])) return trim($_SERVER['HTTP_X_REAL_IP']);
    return $_SERVER['REMOTE_ADDR'] ?? 'unknown_ip';
}
function s4_get_language($default_lang) {
    $supported = ['nb_NO','en_GB','de_DE','cs_CZ','fi_FI'];
    if (isset($_GET['lang']) && in_array($_GET['lang'], $supported)) return $_GET['lang'];
    if (isset($_COOKIE['lang']) && in_array($_COOKIE['lang'], $supported)) return $_COOKIE['lang'];
    if (isset($_SERVER['HTTP_ACCEPT_LANGUAGE'])) {
        preg_match_all('/([a-z]{1,8}(-[a-z]{1,8})?)\s*(;\s*q\s*=\s*(1|0\.[0-9]+))?/i',
            $_SERVER['HTTP_ACCEPT_LANGUAGE'], $m);
        if (count($m[1])) {
            $langs = array_combine($m[1], $m[4]);
            foreach ($langs as $l => $v) { if ($v==='') $langs[$l]=1; }
            arsort($langs, SORT_NUMERIC);
            foreach (array_keys($langs) as $bl) {
                $c = str_replace('-','_',$bl);
                if (in_array($c,$supported)) return $c;
                $s = substr($c,0,2);
                foreach ($supported as $sup) { if (substr($sup,0,2)===$s) return $sup; }
            }
        }
    }
    $cmap = ['NO'=>'nb_NO','SE'=>'nb_NO','DK'=>'nb_NO','GB'=>'en_GB','US'=>'en_GB',
             'CA'=>'en_GB','AU'=>'en_GB','DE'=>'de_DE','AT'=>'de_DE','CH'=>'de_DE',
             'CZ'=>'cs_CZ','SK'=>'cs_CZ','FI'=>'fi_FI'];
    $ip = s4_get_user_ip();
    $gj = @file_get_contents("http://ip-api.com/json/{$ip}?fields=countryCode,status");
    if ($gj) { $g=json_decode($gj); if ($g&&$g->status==='success'&&isset($cmap[$g->countryCode])) return $cmap[$g->countryCode]; }
    return $default_lang;
}

$lang_code = s4_get_language($DEFAULT_LANG);
setcookie('lang', $lang_code, time()+(86400*365), "/");
$lang_short = substr($lang_code, 0, 2);
$t = [];
$lf = $LANG_DIR.'/'.$lang_short.'.json';
if (!file_exists($lf)) $lf = $LANG_DIR.'/'.substr($DEFAULT_LANG,0,2).'.json';
if (file_exists($lf)) { $j=file_get_contents($lf); if ($j) $t=json_decode($j,true); }

// Load station display names from stations.json
$station_display_names = [];
$stations_json_candidates = [
    '/home/httpd/norskmeteornettverk.no/data/stations.json',
    '/home/steinar/norskmeteornettverk.no/data/stations.json',
];
foreach ($stations_json_candidates as $sjf) {
    if (file_exists($sjf)) {
        $sj = json_decode(file_get_contents($sjf), true);
        if ($sj) {
            foreach ($sj as $entry) {
                $stn = $entry['station'] ?? [];
                $n = $stn['name'] ?? '';
                if ($n) $station_display_names[$n] = $stn['display_name'] ?? ucfirst($n);
            }
        }
        break;
    }
}

$a = array_reverse(explode('/', getcwd()));
$path = "/meteor/".$a[1]."/".$a[0]."/";
$date = substr_replace($a[1],'-',4,0); $date = substr_replace($date,'-',7,0);
$time = substr_replace($a[0],':',2,0); $time = substr_replace($time,':',5,0);
$time = preg_replace("/[a-z]/","",$time);
$fp = ($lang_short==='nb') ? '' : $lang_short.'_';

function s4_ap($s,$d) { return file_exists($s)?$s:(file_exists($d)?$d:null); }
$map_jpg     = s4_ap("{$fp}map.jpg",    "map.jpg");
$map_html    = s4_ap("{$fp}map.html",   "map.html");
$height_jpg  = s4_ap("{$fp}height.jpg", "height.jpg");
$orbit_jpg   = s4_ap("{$fp}orbit.jpg",  "orbit.jpg");
$orbit_html  = s4_ap("{$fp}orbit.html", "orbit.html");
$tables_html = s4_ap("{$fp}tables.html",   "tables.html");
$stations_html = s4_ap("{$fp}stations.html","stations.html");
$posvstime_jpg = s4_ap("{$fp}posvstime.jpg","posvstime.jpg");
$spd_acc_jpg   = s4_ap("{$fp}spd_acc.jpg",  "spd_acc.jpg");
$wind_jpg      = s4_ap("{$fp}wind_profile.jpg","wind_profile.jpg");
$og_image      = s4_ap("{$fp}image.jpg","image.jpg");

// ---- Collect station cameras ----
$station_cams = [];
foreach (glob('*/') as $stdir) {
    foreach (glob($stdir.'*/') as $camdir) {
        $camdir = rtrim($camdir,'/');
        $parts = explode('/',$camdir);
        $station=$parts[0]; $cam=$parts[1];

        // Best static preview: gnomonic-grid > gnomonic > grid > plain jpg > fireball.jpg
        $thumb = null;
        $candidates = [
            '*-gnomonic-grid.jpg','*-gnomonic.jpg',
            '*-grid.jpg','*.jpg',
        ];
        foreach ($candidates as $pat) {
            $hits = array_filter(glob($camdir.'/'.$pat), fn($f) => basename($f)!=='fireball.jpg' && basename($f)!=='fireball_orig.jpg');
            if ($hits) { $thumb = array_values($hits)[0]; break; }
        }
        if (!$thumb && file_exists($camdir.'/fireball.jpg')) $thumb = $camdir.'/fireball.jpg';

        // Brightness plot (language-aware)
        $bp = ($lang_short==='nb') ? '' : $lang_short.'_';
        $bpath = $camdir.'/'.$bp.'brightness.jpg';
        if (!file_exists($bpath)) $bpath = $camdir.'/brightness.jpg';
        $brightness = file_exists($bpath) ? $bpath : null;

        // Webm / fireball video
        $webm = file_exists($camdir.'/fireball_neg.webm') ? $camdir.'/fireball_neg.webm' : null;
        $webm_orig = file_exists($camdir.'/fireball_orig.webm') ? $camdir.'/fireball_orig.webm' : null;

        // Station label
        $cam_num = preg_replace('/^cam/i', '', $cam);
        $et = $camdir.'/event.txt';
        $cfg = file_exists($et) ? @parse_ini_file($et, true) : [];
        $stn_name = trim($cfg['station']['name'] ?? '');
        $stn_code = trim($cfg['station']['code'] ?? '');
        $lookup_name = $stn_name ?: $station;
        $stn_display = $station_display_names[$lookup_name] ?? ucfirst($lookup_name);
        if ($stn_display) {
            $label = $stn_display;
            if ($stn_code) $label .= ' ('.$stn_code.')';
            $label .= ' – '.$cam_num;
        } else {
            $label = ucfirst($station).' '.$cam_num;
        }

        if ($thumb || $webm) {
            $station_cams[] = compact('station','cam','camdir','thumb','brightness','webm','webm_orig','label');
        }
    }
}

$is_multi = count($station_cams) > 1;

// Page title string
function s4_title($location, $lang_short, $t, $LANG_DIR) {
    if ($lang_short==='cs') {
        $dec=null; $dp=$LANG_DIR.'/cs_declensions.json';
        if (file_exists($dp)){$d=json_decode(file_get_contents($dp),true);if(isset($d[$location]))$dec=$d[$location];}
        return $dec ? htmlspecialchars(($t['meteor_over']??'Meteor nad').' '.$dec)
                    : htmlspecialchars(($t['meteor']??'Meteor').', '.($t['location']??'poloha').': '.$location);
    } elseif ($lang_short==='fi') {
        $dec=null; $dp=$LANG_DIR.'/fi_declensions.json';
        if (file_exists($dp)){$d=json_decode(file_get_contents($dp),true);if(isset($d[$location]))$dec=$d[$location];}
        if ($dec===null){$last=substr($location,-1);$dec=in_array(mb_strtolower($last,'UTF-8'),['a','e','i','o','u','y','ä','ö'])?$location.'n':$location.'in';}
        return htmlspecialchars(($t['meteor_over']??'Meteor').' '.$dec.' yllä');
    } else {
        return htmlspecialchars(($t['meteor_over']??'Meteor over').' '.$location);
    }
}
$title_str = (file_exists('location.txt') && filesize('location.txt')>0)
    ? s4_title(trim(file_get_contents('location.txt')), $lang_short, $t, $LANG_DIR)
    : htmlspecialchars($t['report_title']??'Meteor Report');
?>
<!DOCTYPE html>
<html lang="<?php echo htmlspecialchars($lang_short); ?>">
<head>
  <meta charset="UTF-8">
  <title><?php echo htmlspecialchars($t['report_title']??'Meteor Report'); ?></title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
<?php if ($og_image): ?><meta property="og:image" content="<?php echo $path.$og_image; ?>"><?php endif; ?>
  <meta property="og:type" content="article">
  <meta property="og:site_name" content="Norsk meteornettverk">
  <meta property="og:url" content="<?php echo $path; ?>">
<style>
:root {
  --primary: #082060; --accent: #c01010; --bg: #f4f6f9;
  --card: #fff; --border: #dee2e6; --text: #333; --muted: #666;
  --font-head: 'Segoe UI','Helvetica Neue',Arial,sans-serif;
  --font-body: Verdana,sans-serif;
}
* { box-sizing: border-box; }
body { font-family: var(--font-body); font-size: 16px; margin: 0; padding: 0;
  background: var(--bg); color: var(--text); line-height: 1.6; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.page-wrapper { max-width: 1400px; margin: 0 auto; padding: 1em; position: relative; }

/* Lang switcher */
.lang-sw { position: absolute; top: 1em; right: 1em; font-size: 1.4em; z-index: 100; }
.lang-sw a { opacity: 0.65; margin: 0 0.15em; text-decoration: none; transition: opacity .2s; }
.lang-sw a:hover { opacity: 1; }

/* Title */
h1.page-title { color: var(--primary); font-family: var(--font-head); font-size: 2.2em;
  font-weight: 300; letter-spacing: -1px; text-align: center; margin: 0.5em 0 0.8em; }

/* =========================================================
   MULTI-STATION LAYOUT
   ========================================================= */

/* -- Thumbnail strip -- */
.thumb-strip { display: flex; flex-wrap: wrap; gap: 0.8em; margin-bottom: 1.5em; }
.thumb-card { flex: 1 1 200px; max-width: 280px; background: var(--card);
  border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.08); overflow: hidden;
  cursor: pointer; transition: box-shadow .3s; }
.thumb-card:hover { box-shadow: 0 8px 24px rgba(0,0,0,.22); }
.thumb-card.active { outline: 3px solid var(--primary); }
/* Image area with zoom — flush against card top, no independent rounding */
.thumb-card .tc-img-wrap { position: relative; width: 100%; height: 150px; overflow: hidden;
  border-radius: 8px 8px 0 0; }
.thumb-card .tc-img-wrap img { width: 100%; height: 100%; object-fit: cover; display: block;
  background: #000; transition: transform .35s ease; transform-origin: center center;
  border-radius: 0 !important; box-shadow: none !important; margin: 0 !important; }
.thumb-card:hover .tc-img-wrap img { transform: scale(2.0); }
/* Static label below image */
.thumb-card .tc-label { font-size: 0.82em; font-family: var(--font-head); color: var(--primary);
  padding: 0.4em 0.6em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* -- Scientific section -- */
.sci-section { margin-bottom: 2em; }
.row { display: flex; flex-direction: row; gap: 1.2em; margin-bottom: 1.2em; align-items: flex-start; }
.row.row-single { max-width: 50%; }
.col { flex: 1; background: var(--card); border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,.06); padding: 1.2em; }
@media (max-width: 800px) { .row.row-single { max-width: 100%; } }
iframe { width: 100%; min-height: 500px; border: none; border-radius: 6px; display: block; }
img.plot { max-width: 100%; height: auto; display: block; margin: 0 auto; border-radius: 6px; }
p.caption { text-align: center; font-size: 0.88em; margin: 0.5em 0 0; color: var(--muted); }

/* -- Station detail tabs -- */
.tab-bar { display: flex; flex-wrap: wrap; gap: 0.3em; padding: 0 0.2em;
  margin-bottom: 0; align-items: flex-end; }
.tab-btn { background: #e4e8f0; border: 1px solid var(--border); border-bottom: none;
  cursor: pointer; font-family: var(--font-head); font-size: 0.92em; color: var(--muted);
  padding: 0.5em 1.3em; border-radius: 6px 6px 0 0; margin-bottom: 0;
  transition: background .15s, color .15s; white-space: nowrap;
  position: relative; top: 1px; }
.tab-btn:hover { background: #f0f3f8; color: var(--primary); }
.tab-btn.active { background: var(--card); color: var(--primary); font-weight: 600;
  border-color: var(--border); border-bottom-color: var(--card);
  box-shadow: 0 -2px 0 0 var(--primary); }
.tab-panel-wrap { border: 1px solid var(--border); border-radius: 0 6px 6px 6px;
  background: var(--card); padding: 1.2em; margin-bottom: 1.5em;
  box-shadow: 0 2px 8px rgba(0,0,0,.06); }
.tab-pane { display: none; }
.tab-pane.active { display: block; }

/* Station detail: image left, links table right */
.stn-detail { display: flex; gap: 1.5em; align-items: flex-start; }
.stn-detail .stn-img { flex: 3; min-width: 0; }
.stn-detail .stn-img img { width: 100%; height: auto; display: block;
  border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.1); }
.stn-detail .stn-img video { width: 100%; height: auto; display: block;
  border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.1); }
.stn-detail .stn-right { flex: 2; min-width: 180px; display: flex; flex-direction: column; gap: 1em; }
.links-card { background: var(--card); border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,.06); padding: 1em; }
.links-card h3 { margin: 0 0 0.5em; font-size: 0.95em; font-family: var(--font-head);
  color: var(--primary); border-bottom: 1px solid var(--border); padding-bottom: 0.3em; }
.links-card ul { margin: 0; padding: 0 0 0 1.1em; }
.links-card ul li { font-size: 0.9em; line-height: 1.8; }
.brightness-card { background: var(--card); border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,.06); overflow: hidden; }
.brightness-card img { width: 100%; height: auto; display: block; }

/* Tables from tables.html */
table.data-table { width: 100%; border-collapse: collapse; }
table.data-table td { padding: 0.5em 0.7em; border-bottom: 1px solid var(--border); text-align: left; }
table.data-table td:first-child { font-weight: bold; color: var(--primary); }
/* stations.html legacy tables */
.col table { width: 100%; border-collapse: collapse; }
.col table td { padding: 0.45em 0.65em; border-bottom: 1px solid var(--border); }
.col > table td:first-child { font-weight: bold; color: var(--primary); }
/* tables.html: both section headings (<b> directly in outer wrapper td) */
.col > table > tbody > tr > td > b { color: var(--primary); }
table img { width: 100% !important; height: auto; }

p { text-align: justify; }
img { max-width: 100%; height: auto; display: block; margin: 1em auto;
  border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,.1); }
.text-center { text-align: center; }
@media (max-width: 1024px) { iframe { min-height: 512px; height: 512px; } }
table img { width: 100% !important; height: auto; }

/* Shared */
hr { border: none; border-top: 1px solid var(--border); margin: 1.5em 0; }
footer { text-align: center; color: var(--muted); font-size: 0.9em; margin-top: 2em; padding: 1em 0; }

@media (max-width: 800px) {
  .row { flex-direction: column; }
  .stn-detail { flex-direction: column; }
  .thumb-strip { gap: 0.5em; }
  .thumb-card { flex: 1 1 45%; max-width: none; }
  /* Mobile modal sizing */
  .modal-content{min-width:0!important;width:95vw!important;min-height:auto!important;max-height:98vh}
  .modal-backdrop{padding:10px}
  /* Mobile video sizing - fill without letterboxing */
  .media-wrapper{height:auto!important;width:100%!important;background:transparent!important}
  .media-wrapper img,.media-wrapper video{width:100%!important;height:100%!important;max-height:none!important;display:block!important;object-fit:cover!important}
}
.modal-backdrop{position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:1000;display:flex;align-items:center;justify-content:center;padding:20px}
.modal-content{background:#0a1628;border-radius:12px;max-width:95vw;max-height:95vh;min-width:800px;min-height:600px;display:flex;flex-direction:column;overflow:hidden;position:relative;color:#e0e6ed}
.modal-header{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid #1a3a5c;background:#0d1f35;position:relative;z-index:20}
.modal-header h3{margin:0;font-size:16px;color:#e0e6ed}
.modal-close{font-size:24px;background:none;border:none;cursor:pointer;color:#8aa4be;padding:0 4px;line-height:1;position:relative;z-index:21}
.modal-close:hover{color:#fff}
.media-wrapper{position:relative;width:100%;height:auto;max-height:75vh;background:#050d18;display:flex;align-items:center;justify-content:center;overflow:hidden;z-index:1}
.media-wrapper img,.media-wrapper video{width:100%;height:auto;max-height:75vh;display:block;object-fit:cover}
/* Hide browser native video controls */
video::-webkit-media-controls{display:none!important}
video::-webkit-media-controls-enclosure{display:none!important}
video::-webkit-media-controls-panel{display:none!important}
video::-webkit-media-controls-play-button{display:none!important}
video::-webkit-media-controls-timeline{display:none!important}
video::-webkit-media-controls-current-time-display{display:none!important}
video::-webkit-media-controls-time-remaining-display{display:none!important}
video::-webkit-media-controls-volume-control-container{display:none!important}
video::-webkit-media-controls-fullscreen-button{display:none!important}
/* Ensure panZoomContainer clips transformed content */
.panZoomContainer{overflow:hidden;width:100%;height:100%;position:relative;clip-path:inset(0);contain:layout paint}
/* Fullscreen mode - fill entire screen */
.media-wrapper:fullscreen{position:fixed;inset:0;width:100vw;height:100vh;max-height:none;padding:0}
.media-wrapper:fullscreen:has(video){aspect-ratio:auto;height:100vh}
.media-wrapper:fullscreen img,.media-wrapper:fullscreen video{max-width:none;max-height:none;width:100vw;height:100vh;object-fit:cover}
.media-wrapper:fullscreen{clip-path:none!important;z-index:auto!important;contain:none!important}
.media-wrapper:fullscreen .panZoomContainer{clip-path:none!important;contain:none!important}
.panZoomContainer:fullscreen{position:fixed;inset:0;width:100vw;height:100vh;background:#000;max-height:none!important}
.panZoomContainer:fullscreen video,.panZoomContainer:fullscreen img{width:100vw!important;height:100vh!important;max-width:none!important;max-height:none!important;object-fit:cover}
.media-controls{display:flex;gap:6px;padding:10px 14px;background:#0d1f35;border-top:1px solid #1a3a5c;flex-wrap:wrap;align-items:center;position:relative;z-index:100;font-size:11px}
.media-btn{padding:5px 12px;background:#1e4d7b;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:11px;font-family:inherit}
.media-btn:hover{background:#2a6aa3}
.frame-btn{background:transparent;border:none;padding:4px 8px;font-size:16px;color:#e0e6ed;cursor:pointer;line-height:1}
.frame-btn:hover{color:#4a9fd4}
.zoom-slider{width:100px;accent-color:#4a9fd4}
.zoom-label{font-size:11px;color:#8aa4be;margin-right:4px}
.media-btn:disabled{background:#1a3a5c;cursor:not-allowed;color:#5a7a9a}
.toggle-group{display:flex;gap:4px;align-items:center;margin-right:12px}
.toggle-label{font-size:12px;color:#8aa4be}
/* Text file viewer */
.text-viewer-content{background:#0a1628;color:#d4d4d4;padding:20px;max-height:70vh;overflow:auto;font-family:'Consolas','Monaco','Courier New',monospace;font-size:13px;line-height:1.5}
.text-viewer-pre{margin:0;white-space:pre-wrap;word-wrap:break-word}
.text-section{color:#4a9fd4;font-weight:bold;margin-top:10px}  /* [section] headers in lighter blue */
.text-key{color:#8aa4be}  /* keys in muted blue */
.text-value{color:#d4a574}  /* values in warm orange */
.text-number{color:#7ec4a0}  /* numbers in mint green */
.text-comment{color:#6a9955;font-style:italic}  /* comments in green */
.text-timestamp{color:#dcdcaa}  /* timestamps in yellow */
.text-row:nth-child(even){background:#0d1f35}  /* alternating row colors in dark blue */
.text-header{background:#1a3a5c;color:#fff;padding:5px 10px;font-weight:bold}
</style>
</head>
<body>
<div class="page-wrapper">

  <div class="lang-sw">
    <a href="?lang=nb_NO" title="Norsk">🇳🇴</a>
    <a href="?lang=en_GB" title="English">🇬🇧</a>
    <a href="?lang=de_DE" title="Deutsch">🇩🇪</a>
    <a href="?lang=cs_CZ" title="Čeština">🇨🇿</a>
    <a href="?lang=fi_FI" title="Suomi">🇫🇮</a>
  </div>

  <h1 class="page-title">
    <?php echo $title_str; ?>
    <span style="white-space:nowrap"><?php echo $date; ?></span>
    <?php echo $time; ?> <?php echo htmlspecialchars($t['utc']??'UTC'); ?>
  </h1>

<?php if ($is_multi): ?>
  <!-- =====================================================
       MULTI-STATION LAYOUT
       ===================================================== -->

  <!-- 1. Thumbnail strip -->
  <div class="thumb-strip">
    <?php foreach ($station_cams as $i => $sc): ?>
    <div class="thumb-card<?php echo $i===0?' active':''; ?>" onclick="showStn(<?php echo $i; ?>, true)" id="tc-<?php echo $i; ?>">
      <div class="tc-img-wrap">
        <?php if ($sc['thumb']): ?>
          <img src="<?php echo htmlspecialchars($sc['thumb']); ?>" alt="<?php echo htmlspecialchars($sc['label']); ?>">
        <?php else: ?>
          <div style="width:100%;height:150px;background:#333;"></div>
        <?php endif; ?>
      </div>
      <div class="tc-label"><?php echo htmlspecialchars($sc['label']); ?></div>
    </div>
    <?php endforeach; ?>
  </div>

  <!-- 2. Scientific data -->
  <div class="sci-section">

    <?php if ($map_html || $map_jpg): ?>
    <div class="row">
      <?php if ($map_html && $map_jpg): ?>
        <div class="col">
          <script>document.addEventListener('DOMContentLoaded',function(){var p=document.getElementById('s4mp');if(p){var f=document.createElement('iframe');f.src='<?php echo $map_html;?>';p.parentNode.replaceChild(f,p);}});</script>
          <div id="s4mp"></div>
          <p class="caption">
            <a href="obs_<?php echo $date;?>_<?php echo $time;?>.kml"><?php echo htmlspecialchars($t['kml_file']??'KML'); ?></a>
            &nbsp;|&nbsp;<a href="<?php echo $map_html;?>"><?php echo htmlspecialchars($t['interactive_map']??'Interactive map'); ?></a>
          </p>
        </div>
        <div class="col">
          <img class="plot" src="<?php echo $map_jpg;?>" alt="map">
          <p class="caption"><?php echo htmlspecialchars($t['map_caption']??''); ?></p>
        </div>
      <?php elseif ($map_html): ?>
        <div class="col">
          <script>document.addEventListener('DOMContentLoaded',function(){var p=document.getElementById('s4mp');if(p){var f=document.createElement('iframe');f.src='<?php echo $map_html;?>';p.parentNode.replaceChild(f,p);}});</script>
          <div id="s4mp"></div>
          <p class="caption">
            <a href="obs_<?php echo $date;?>_<?php echo $time;?>.kml"><?php echo htmlspecialchars($t['kml_file']??'KML'); ?></a>
            &nbsp;|&nbsp;<a href="<?php echo $map_html;?>"><?php echo htmlspecialchars($t['interactive_map']??'Interactive map'); ?></a>
          </p>
        </div>
      <?php else: ?>
        <div class="col">
          <img class="plot" src="<?php echo $map_jpg;?>" alt="map">
          <p class="caption"><a href="obs_<?php echo $date;?>_<?php echo $time;?>.kml"><?php echo htmlspecialchars($t['kml_file']??'KML'); ?></a></p>
        </div>
      <?php endif; ?>
    </div>
    <?php endif; ?>

    <?php if ($height_jpg || $wind_jpg): ?>
    <?php $both_hw = $height_jpg && $wind_jpg; ?>
    <div class="row<?php echo $both_hw ? '' : ' row-single'; ?>">
      <?php if ($height_jpg): ?><div class="col"><img class="plot" src="<?php echo $height_jpg;?>" alt="height profile"></div><?php endif; ?>
      <?php if ($wind_jpg): ?><div class="col"><img class="plot" src="<?php echo $wind_jpg;?>" alt="wind profile"></div><?php endif; ?>
    </div>
    <?php endif; ?>

    <?php if ($orbit_jpg || $orbit_html || $tables_html): ?>
    <div class="row">
      <?php if ($tables_html): ?><div class="col"><?php include $tables_html; ?></div><?php endif; ?>
      <?php if ($orbit_jpg || $orbit_html): ?>
      <div class="col">
        <div id="s4op"><?php if ($orbit_jpg) echo "<img class='plot' src='{$orbit_jpg}' alt='orbit'>"; ?></div>
        <?php if ($orbit_html):
          echo "<script>document.addEventListener('DOMContentLoaded',function(){var p=document.getElementById('s4op');if(p){var f=document.createElement('iframe');f.src='{$orbit_html}';p.parentNode.replaceChild(f,p);}});</script>";
          echo "<p class='caption'>";
          if ($orbit_jpg) echo "<a href='{$orbit_jpg}'>".htmlspecialchars($t['non_interactive_plot']??'Static')."</a> | ";
          echo "<a href='{$orbit_html}'>".htmlspecialchars($t['interactive_plot']??'Interactive')."</a></p>";
        endif; ?>
      </div>
      <?php endif; ?>
    </div>
    <?php endif; ?>

    <?php if ($posvstime_jpg || $spd_acc_jpg): ?>
    <div class="row">
      <?php if ($posvstime_jpg): ?><div class="col"><img class="plot" src="<?php echo $posvstime_jpg;?>" alt="position vs time"></div><?php endif; ?>
      <?php if ($spd_acc_jpg): ?><div class="col"><img class="plot" src="<?php echo $spd_acc_jpg;?>" alt="speed & acceleration"></div><?php endif; ?>
    </div>
    <?php endif; ?>

  </div><!-- /sci-section -->

  <!-- 3. Tabbed station detail -->
  <hr>
  <div id="stn-tabs-anchor"></div>
  <div class="tab-bar">
    <?php foreach ($station_cams as $i => $sc): ?>
      <button class="tab-btn<?php echo $i===0?' active':''; ?>" onclick="showStn(<?php echo $i; ?>)" id="tb-<?php echo $i; ?>"><?php echo htmlspecialchars($sc['label']); ?></button>
    <?php endforeach; ?>
  </div>
  <div class="tab-panel-wrap">

  <?php foreach ($station_cams as $i => $sc): ?>
  <div class="tab-pane<?php echo $i===0?' active':''; ?>" id="tp-<?php echo $i; ?>">
    <div class="stn-detail">

      <!-- Left: best available image (static, gnomonic preferred) -->
      <div class="stn-img">
        <?php if ($sc['thumb']): ?>
          <img src="<?php echo htmlspecialchars($sc['thumb']); ?>" alt="<?php echo htmlspecialchars($sc['label']); ?>">
        <?php elseif ($sc['webm']): ?>
          <video autoplay loop muted playsinline>
            <source src="<?php echo htmlspecialchars($sc['webm']); ?>" type="video/webm">
          </video>
        <?php endif; ?>
      </div>

      <!-- Right: links table + brightness plot -->
      <div class="stn-right">
        <?php
        // Regenerate the links exactly as stations.html does, but rendered into our card layout.
        // We read the camdir file listing to build the link groups dynamically.
        $cd = $sc['camdir'];
        $pfx_glob = glob($cd.'/*-2*.mp4'); // files with a timestamp in name
        // Detect the timestamp prefix used in this camdir
        $ts_prefix = null;
        foreach (array_merge(glob($cd.'/*-gnomonic-grid.jpg'), glob($cd.'/*-gnomonic.jpg'), glob($cd.'/*-grid.jpg')) as $f) {
            $bn = basename($f);
            // strip suffix after last known pattern
            if (preg_match('/^(.+?)-gnomonic-grid\.jpg$/', $bn, $mm)) { $ts_prefix = $mm[1]; break; }
            if (preg_match('/^(.+?)-gnomonic\.jpg$/',      $bn, $mm)) { $ts_prefix = $mm[1]; break; }
            if (preg_match('/^(.+?)-grid\.jpg$/',          $bn, $mm)) { $ts_prefix = $mm[1]; break; }
        }
        if (!$ts_prefix) {
            foreach (glob($cd.'/*[0-9].jpg') as $f) {
                $bn = basename($f, '.jpg');
                if (!preg_match('/fireball/', $bn)) { $ts_prefix = $bn; break; }
            }
        }
        $base_url = "/meteor/{$a[1]}/{$a[0]}/{$cd}";

        // Video links
        $vids = [];
        $vid_defs = [
            $ts_prefix.'-gnomonic.mp4'           => $t['gnomonic']??'Gnomonisk',
            $ts_prefix.'-gnomonic-grid.mp4'      => $t['gnomonic_with_coords']??'Gnomonisk med koordinater',
            $ts_prefix.'-orig.mp4'               => $t['original']??'Original',
            $ts_prefix.'.mp4'                    => $t['original']??'Original',
            $ts_prefix.'-grid.mp4'               => $t['original_with_coords']??'Original med koordinater',
        ];
        $orig_done = false;
        foreach ($vid_defs as $fname => $lbl) {
            if (str_ends_with($fname,'-orig.mp4') || str_ends_with($fname,'.mp4')) {
                if ($orig_done) continue;
            }
            if ($ts_prefix && file_exists($cd.'/'.$fname)) {
                $vids[$fname] = $lbl;
                if (str_ends_with($fname,'-orig.mp4') || ($fname===$ts_prefix.'.mp4')) $orig_done=true;
            }
        }

        // Image links
        $imgs = [];
        $img_defs = [
            $ts_prefix.'-gnomonic.jpg'            => $t['gnomonic']??'Gnomonisk',
            $ts_prefix.'-gnomonic-grid.jpg'       => $t['gnomonic_with_coords']??'Gnomonisk med koordinater',
            $ts_prefix.'-gnomonic-grid-uncorr.jpg'=> $t['gnomonic_uncorrected_with_coords']??'Ukorrigert gnomonisk med koordinater',
            $ts_prefix.'-gnomonic-labels.jpg'     => $t['gnomonic_with_labels']??'Gnomonisk med annotering',
            $ts_prefix.'-gnomonic-labels-uncorr.jpg' => $t['gnomonic_uncorrected_with_labels']??'Ukorrigert gnomonisk med annotering',
            $ts_prefix.'.jpg'                     => $t['original']??'Original',
            $ts_prefix.'-grid.jpg'                => $t['original_with_coords']??'Original med koordinater',
            $ts_prefix.'-mask.jpg'                => $t['original_with_mask']??'Original med maske',
        ];
        foreach ($img_defs as $fname => $lbl) {
            if ($ts_prefix && file_exists($cd.'/'.$fname)) $imgs[$fname] = $lbl;
        }

        // Text file links
        $txts = [];
        $txt_defs = [
            'event.txt'                  => $t['detection']??'Deteksjon',
            $ts_prefix.'.txt'            => $t['observation']??'Observasjon',
            'centroid2.txt'              => $t['coordinates']??'Koordinater',
            'stderr.txt'                 => $t['error_messages']??'Feilmeldinger',
            'report.log'                 => $t['log_file']??'Logg',
        ];
        foreach ($txt_defs as $fname => $lbl) {
            if ($ts_prefix && file_exists($cd.'/'.$fname)) $txts[$fname] = $lbl;
            elseif (!$ts_prefix && file_exists($cd.'/'.$fname)) $txts[$fname] = $lbl;
        }
        ?>

        <?php if (!empty($vids)): ?>
        <div class="links-card">
          <h3><?php echo htmlspecialchars($t['videos']??'Videoer'); ?></h3>
          <ul><?php
            $allVidVariants = [];
            foreach ($vids as $vf => $vl) {
                $allVidVariants[] = ['url' => "{$base_url}/{$vf}", 'label' => $vl, 'desc' => $vl];
            }
            foreach ($vids as $fname => $lbl): ?>
              <li><a href="#" onclick="openMediaPlayer('<?php echo "{$base_url}/{$fname}"; ?>', 'video', <?php echo htmlspecialchars(json_encode($allVidVariants, JSON_HEX_QUOT|JSON_HEX_TAG|JSON_HEX_AMP)); ?>, '<?php echo htmlspecialchars($sc['label'] ?? ''); ?>'); return false;"><?php echo htmlspecialchars($lbl); ?></a></li>
            <?php endforeach; ?>
          </ul>
        </div>
        <?php endif; ?>

        <?php if (!empty($imgs)): ?>
        <div class="links-card">
          <h3><?php echo htmlspecialchars($t['images']??'Bilder'); ?></h3>
          <ul><?php
            $allImgVariants = [];
            foreach ($imgs as $if => $il) {
                $allImgVariants[] = ['url' => "{$base_url}/{$if}", 'label' => $il, 'desc' => $il];
            }
            foreach ($imgs as $fname => $lbl): ?>
              <li><a href="#" onclick="openMediaPlayer('<?php echo "{$base_url}/{$fname}"; ?>', 'image', <?php echo htmlspecialchars(json_encode($allImgVariants, JSON_HEX_QUOT|JSON_HEX_TAG|JSON_HEX_AMP)); ?>, '<?php echo htmlspecialchars($sc['label'] ?? ''); ?>'); return false;"><?php echo htmlspecialchars($lbl); ?></a></li>
            <?php endforeach; ?>
          </ul>
        </div>
        <?php endif; ?>

        <?php if (!empty($txts)): ?>
        <div class="links-card">
          <h3><?php echo htmlspecialchars($t['text_files']??'Tekstfiler'); ?></h3>
          <ul>
            <?php foreach ($txts as $fname => $lbl): ?>
              <li><a href="#" onclick="openTextViewer('<?php echo "{$base_url}/{$fname}"; ?>', '<?php echo htmlspecialchars($lbl); ?>'); return false;"><?php echo htmlspecialchars($lbl); ?></a></li>
            <?php endforeach; ?>
          </ul>
        </div>
        <?php endif; ?>

        <?php if ($sc['brightness']): ?>
        <div class="brightness-card">
          <img src="<?php echo htmlspecialchars($sc['brightness']); ?>" alt="brightness">
        </div>
        <?php endif; ?>

      </div><!-- /stn-right -->
    </div><!-- /stn-detail -->
  </div><!-- /tab-pane -->
  <?php endforeach; ?>

  </div><!-- /tab-panel-wrap -->

<?php else: ?>
  <!-- =====================================================
       SINGLE-STATION LAYOUT — same style as multi-station
       ===================================================== -->

  <!-- Scientific plots (reuse same .sci-section / .row / .col markup) -->
  <div class="sci-section">

    <?php if ($map_html || $map_jpg): ?>
    <div class="row">
      <?php if ($map_html && $map_jpg): ?>
        <div class="col">
          <script>document.addEventListener('DOMContentLoaded',function(){var p=document.getElementById('s4smp');if(p){var f=document.createElement('iframe');f.src='<?php echo $map_html;?>';p.parentNode.replaceChild(f,p);}});</script>
          <div id="s4smp"></div>
          <p class="caption">
            <a href="obs_<?php echo $date;?>_<?php echo $time;?>.kml"><?php echo htmlspecialchars($t['kml_file']??'KML'); ?></a>
            &nbsp;|&nbsp;<a href="<?php echo $map_html;?>"><?php echo htmlspecialchars($t['interactive_map']??'Interactive map'); ?></a>
          </p>
        </div>
        <div class="col">
          <img class="plot" src="<?php echo $map_jpg;?>" alt="map">
          <p class="caption"><?php echo htmlspecialchars($t['map_caption']??''); ?></p>
        </div>
      <?php elseif ($map_html): ?>
        <div class="col">
          <script>document.addEventListener('DOMContentLoaded',function(){var p=document.getElementById('s4smp');if(p){var f=document.createElement('iframe');f.src='<?php echo $map_html;?>';p.parentNode.replaceChild(f,p);}});</script>
          <div id="s4smp"></div>
          <p class="caption">
            <a href="obs_<?php echo $date;?>_<?php echo $time;?>.kml"><?php echo htmlspecialchars($t['kml_file']??'KML'); ?></a>
            &nbsp;|&nbsp;<a href="<?php echo $map_html;?>"><?php echo htmlspecialchars($t['interactive_map']??'Interactive map'); ?></a>
          </p>
        </div>
      <?php else: ?>
        <div class="col">
          <img class="plot" src="<?php echo $map_jpg;?>" alt="map">
          <p class="caption"><a href="obs_<?php echo $date;?>_<?php echo $time;?>.kml"><?php echo htmlspecialchars($t['kml_file']??'KML'); ?></a></p>
        </div>
      <?php endif; ?>
    </div>
    <?php endif; ?>

    <?php if ($height_jpg || $wind_jpg): ?>
    <?php $both_hw = $height_jpg && $wind_jpg; ?>
    <div class="row<?php echo $both_hw ? '' : ' row-single'; ?>">
      <?php if ($height_jpg): ?><div class="col"><img class="plot" src="<?php echo $height_jpg;?>" alt="height profile"></div><?php endif; ?>
      <?php if ($wind_jpg): ?><div class="col"><img class="plot" src="<?php echo $wind_jpg;?>" alt="wind profile"></div><?php endif; ?>
    </div>
    <?php endif; ?>

    <?php if ($orbit_jpg || $orbit_html || $tables_html): ?>
    <div class="row">
      <?php if ($tables_html): ?><div class="col"><?php include $tables_html; ?></div><?php endif; ?>
      <?php if ($orbit_jpg || $orbit_html): ?>
      <div class="col">
        <div id="s4sop"><?php if ($orbit_jpg) echo "<img class='plot' src='{$orbit_jpg}' alt='orbit'>"; ?></div>
        <?php if ($orbit_html):
          echo "<script>document.addEventListener('DOMContentLoaded',function(){var p=document.getElementById('s4sop');if(p){var f=document.createElement('iframe');f.src='{$orbit_html}';p.parentNode.replaceChild(f,p);}});</script>";
          echo "<p class='caption'>";
          if ($orbit_jpg) echo "<a href='{$orbit_jpg}'>".htmlspecialchars($t['non_interactive_plot']??'Static')."</a> | ";
          echo "<a href='{$orbit_html}'>".htmlspecialchars($t['interactive_plot']??'Interactive')."</a></p>";
        endif; ?>
      </div>
      <?php endif; ?>
    </div>
    <?php endif; ?>

    <?php if ($posvstime_jpg || $spd_acc_jpg): ?>
    <div class="row">
      <?php if ($posvstime_jpg): ?><div class="col"><img class="plot" src="<?php echo $posvstime_jpg;?>" alt="position vs time"></div><?php endif; ?>
      <?php if ($spd_acc_jpg): ?><div class="col"><img class="plot" src="<?php echo $spd_acc_jpg;?>" alt="speed & acceleration"></div><?php endif; ?>
    </div>
    <?php endif; ?>

  </div><!-- /sci-section -->

  <!-- Single station detail panel — same stn-detail layout as multi-station, no tab bar -->
  <?php if (!empty($station_cams)): $sc = $station_cams[0]; ?>
  <hr>
  <div class="tab-panel-wrap single">
    <div class="stn-detail">

      <div class="stn-img">
        <?php if ($sc['thumb']): ?>
          <img src="<?php echo htmlspecialchars($sc['thumb']); ?>" alt="<?php echo htmlspecialchars($sc['label']); ?>">
        <?php elseif ($sc['webm']): ?>
          <video autoplay loop muted playsinline>
            <source src="<?php echo htmlspecialchars($sc['webm']); ?>" type="video/webm">
          </video>
        <?php endif; ?>
      </div>

      <div class="stn-right">
        <?php
        $cd = $sc['camdir'];
        $ts_prefix = null;
        foreach (array_merge(glob($cd.'/*-gnomonic-grid.jpg')?:[], glob($cd.'/*-gnomonic.jpg')?:[], glob($cd.'/*-grid.jpg')?:[]) as $f) {
            $bn = basename($f);
            if (preg_match('/^(.+?)-gnomonic-grid\.jpg$/', $bn, $mm)) { $ts_prefix = $mm[1]; break; }
            if (preg_match('/^(.+?)-gnomonic\.jpg$/',      $bn, $mm)) { $ts_prefix = $mm[1]; break; }
            if (preg_match('/^(.+?)-grid\.jpg$/',          $bn, $mm)) { $ts_prefix = $mm[1]; break; }
        }
        if (!$ts_prefix) {
            foreach (glob($cd.'/*[0-9].jpg')?:[] as $f) {
                $bn = basename($f, '.jpg');
                if (!preg_match('/fireball/', $bn)) { $ts_prefix = $bn; break; }
            }
        }
        $base_url = "/meteor/{$a[1]}/{$a[0]}/{$cd}";

        $vids = []; $orig_done = false;
        $vid_defs = $ts_prefix ? [
            $ts_prefix.'-gnomonic.mp4'      => $t['gnomonic']??'Gnomonisk',
            $ts_prefix.'-gnomonic-grid.mp4' => $t['gnomonic_with_coords']??'Gnomonisk med koordinater',
            $ts_prefix.'-orig.mp4'          => $t['original']??'Original',
            $ts_prefix.'.mp4'               => $t['original']??'Original',
            $ts_prefix.'-grid.mp4'          => $t['original_with_coords']??'Original med koordinater',
        ] : [];
        foreach ($vid_defs as $fname => $lbl) {
            if ((str_ends_with($fname,'-orig.mp4')||$fname===$ts_prefix.'.mp4') && $orig_done) continue;
            if (file_exists($cd.'/'.$fname)) {
                $vids[$fname] = $lbl;
                if (str_ends_with($fname,'-orig.mp4')||$fname===$ts_prefix.'.mp4') $orig_done=true;
            }
        }
        $imgs = []; $img_defs = $ts_prefix ? [
            $ts_prefix.'-gnomonic.jpg'             => $t['gnomonic']??'Gnomonisk',
            $ts_prefix.'-gnomonic-grid.jpg'        => $t['gnomonic_with_coords']??'Gnomonisk med koordinater',
            $ts_prefix.'-gnomonic-grid-uncorr.jpg' => $t['gnomonic_uncorrected_with_coords']??'Ukorrigert gnomonisk med koordinater',
            $ts_prefix.'-gnomonic-labels.jpg'      => $t['gnomonic_with_labels']??'Gnomonisk med annotering',
            $ts_prefix.'-gnomonic-labels-uncorr.jpg' => $t['gnomonic_uncorrected_with_labels']??'Ukorrigert gnomonisk med annotering',
            $ts_prefix.'.jpg'                      => $t['original']??'Original',
            $ts_prefix.'-grid.jpg'                 => $t['original_with_coords']??'Original med koordinater',
            $ts_prefix.'-mask.jpg'                 => $t['original_with_mask']??'Original med maske',
        ] : [];
        foreach ($img_defs as $fname => $lbl) { if (file_exists($cd.'/'.$fname)) $imgs[$fname]=$lbl; }
        $txts = [];
        foreach ([
            'event.txt'              => $t['detection']??'Deteksjon',
            ($ts_prefix?$ts_prefix.'.txt':'') => $t['observation']??'Observasjon',
            'centroid2.txt'          => $t['coordinates']??'Koordinater',
            'stderr.txt'             => $t['error_messages']??'Feilmeldinger',
            'report.log'             => $t['log_file']??'Logg',
        ] as $fname => $lbl) { if ($fname && file_exists($cd.'/'.$fname)) $txts[$fname]=$lbl; }
        ?>

        <?php if (!empty($vids)): ?>
        <div class="links-card">
          <h3><?php echo htmlspecialchars($t['videos']??'Videoer'); ?></h3>
          <ul><?php 
            // Build all video variants array
            $allVidVariants = [];
            foreach ($vids as $vf => $vl) {
                $allVidVariants[] = ['url' => "{$base_url}/{$vf}", 'label' => $vl, 'desc' => $vl];
            }
            foreach ($vids as $fname => $lbl): ?>
            <li><a href="#" onclick="openMediaPlayer('<?php echo "{$base_url}/{$fname}"; ?>', 'video', <?php echo htmlspecialchars(json_encode($allVidVariants, JSON_HEX_QUOT|JSON_HEX_TAG|JSON_HEX_AMP)); ?>, '<?php echo htmlspecialchars($sc['label'] ?? ''); ?>'); return false;"><?php echo htmlspecialchars($lbl); ?></a></li>
          <?php endforeach; ?></ul>
        </div>
        <?php endif; ?>

        <?php if (!empty($imgs)): ?>
        <div class="links-card">
          <h3><?php echo htmlspecialchars($t['images']??'Bilder'); ?></h3>
          <ul><?php 
            // Build all image variants array
            $allImgVariants = [];
            foreach ($imgs as $if => $il) {
                $allImgVariants[] = ['url' => "{$base_url}/{$if}", 'label' => $il, 'desc' => $il];
            }
            foreach ($imgs as $fname => $lbl): ?>
            <li><a href="#" onclick="openMediaPlayer('<?php echo "{$base_url}/{$fname}"; ?>', 'image', <?php echo htmlspecialchars(json_encode($allImgVariants, JSON_HEX_QUOT|JSON_HEX_TAG|JSON_HEX_AMP)); ?>, '<?php echo htmlspecialchars($sc['label'] ?? ''); ?>'); return false;"><?php echo htmlspecialchars($lbl); ?></a></li>
          <?php endforeach; ?></ul>
        </div>
        <?php endif; ?>

        <?php if (!empty($txts)): ?>
        <div class="links-card">
          <h3><?php echo htmlspecialchars($t['text_files']??'Tekstfiler'); ?></h3>
          <ul><?php foreach ($txts as $fname => $lbl): ?>
            <li><a href="#" onclick="openTextViewer('<?php echo "{$base_url}/{$fname}"; ?>', '<?php echo htmlspecialchars($lbl); ?>'); return false;"><?php echo htmlspecialchars($lbl); ?></a></li>
          <?php endforeach; ?></ul>
        </div>
        <?php endif; ?>

        <?php if ($sc['brightness']): ?>
        <div class="brightness-card">
          <img src="<?php echo htmlspecialchars($sc['brightness']); ?>" alt="brightness">
        </div>
        <?php endif; ?>

      </div><!-- /stn-right -->
    </div><!-- /stn-detail -->
  </div><!-- /tab-panel-wrap -->
  <?php endif; ?>

<?php endif; // end single/multi branch ?>

  <footer><p class="text-center"><?php echo htmlspecialchars($t['footer_text']??'Automatically generated by the Norwegian Meteor Network.'); ?></p></footer>

</div><!-- /page-wrapper -->

<?php if ($is_multi): ?>
<script>
function showStn(i, scroll) {
    // Tab buttons
    document.querySelectorAll('.tab-btn').forEach(function(b){ b.classList.remove('active'); });
    var tb = document.getElementById('tb-'+i); if (tb) tb.classList.add('active');
    // Tab panes
    document.querySelectorAll('.tab-pane').forEach(function(p){ p.classList.remove('active'); });
    var tp = document.getElementById('tp-'+i); if (tp) tp.classList.add('active');
    // Thumbnail strip highlight
    document.querySelectorAll('.thumb-card').forEach(function(c){ c.classList.remove('active'); });
    var tc = document.getElementById('tc-'+i); if (tc) tc.classList.add('active');
    // Scroll to tab panel when triggered from thumbnail
    if (scroll) {
        var anchor = document.getElementById('stn-tabs-anchor');
        if (anchor) anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}
</script>
<?php endif; ?>

<script>
// Video and Image Player
function openMediaPlayer(url, type, variants, stationLabel) {
    const existing = document.getElementById('media-modal');
    if (existing) existing.remove();
    
    const backdrop = document.createElement('div');
    backdrop.id = 'media-modal';
    backdrop.className = 'modal-backdrop';
    backdrop.onclick = function(e) {
        if (e.target === backdrop) closeMediaPlayer();
    };
    
    const modal = document.createElement('div');
    modal.className = 'modal-content';
    
    // Header with station name and version
    const header = document.createElement('div');
    header.className = 'modal-header';
    const title = document.createElement('h3');
    const versionDesc = variants && variants[0] ? (variants[0].desc || variants[0].label) : '';
    title.textContent = (stationLabel || '') + (versionDesc ? ' - ' + versionDesc : '');
    const closeBtn = document.createElement('button');
    closeBtn.className = 'modal-close';
    closeBtn.innerHTML = '&times;';
    closeBtn.onclick = closeMediaPlayer;
    header.appendChild(title);
    header.appendChild(closeBtn);
    
    // Media wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'media-wrapper';
    
    let mediaElement;
    let zoomSlider = null;
    let panZoomContainer = null;
    if (type === 'video') {
        // Video with pan-and-zoom support
        panZoomContainer = document.createElement('div');
        panZoomContainer.className = 'panZoomContainer';
        panZoomContainer.style.overflow = 'hidden';
        panZoomContainer.style.width = '100%';
        panZoomContainer.style.height = '100%';
        panZoomContainer.style.display = 'flex';
        panZoomContainer.style.alignItems = 'center';
        panZoomContainer.style.justifyContent = 'center';
        panZoomContainer.style.position = 'relative';
        panZoomContainer.style.cursor = 'grab';
        
        mediaElement = document.createElement('video');
        mediaElement.src = url;
        mediaElement.controls = false;
        mediaElement.setAttribute('controlsList', 'nodownload nofullscreen noremoteplayback');
        mediaElement.autoplay = true;
        mediaElement.loop = true;
        mediaElement.playsInline = true;
        mediaElement.setAttribute('playsinline', 'true');
        mediaElement.setAttribute('webkit-playsinline', 'true');
        mediaElement.setAttribute('x5-playsinline', 'true');
        mediaElement.setAttribute('x5-video-player-type', 'h5');
        mediaElement.setAttribute('x5-video-player-fullscreen', 'false');
        mediaElement.disablePictureInPicture = true;
        mediaElement.id = 'media-element';
        mediaElement.style.width = '100%';
        mediaElement.style.height = '100%';
        mediaElement.style.objectFit = 'contain';
        mediaElement.style.transition = 'transform 0.1s ease-out';
        mediaElement.style.transformOrigin = 'center center';
        mediaElement.style.cursor = 'grab';
        panZoomContainer.appendChild(mediaElement);
        wrapper.appendChild(panZoomContainer);
        
        // Click to play/pause when not dragging
        mediaElement.addEventListener('click', function(e) {
            if (hasDragged) {
                e.preventDefault();
                e.stopPropagation();
            } else {
                // Toggle play/pause
                if (mediaElement.paused) {
                    mediaElement.play();
                } else {
                    mediaElement.pause();
                }
            }
        });
        
        // Pan and zoom state for video
        let scale = 1;
        let panning = false;
        let pointX = 0;
        let pointY = 0;
        let startX = 0;
        let startY = 0;
        let dragStartX = 0;
        let dragStartY = 0;
        let hasDragged = false;
        
        function clampPosition() {
            if (scale <= 1) {
                pointX = 0;
                pointY = 0;
            } else {
                const rect = panZoomContainer.getBoundingClientRect();
                const vidWidth = rect.width * scale;
                const vidHeight = rect.height * scale;
                const maxX = (vidWidth - rect.width) / 2;
                const maxY = (vidHeight - rect.height) / 2;
                pointX = Math.max(-maxX, Math.min(maxX, pointX));
                pointY = Math.max(-maxY, Math.min(maxY, pointY));
            }
        }
        
        function setTransform() {
            clampPosition();
            mediaElement.style.transform = 'translate(' + pointX + 'px, ' + pointY + 'px) scale(' + scale + ')';
        }
        
        // Mouse wheel zoom (follows cursor)
        panZoomContainer.addEventListener('wheel', function(e) {
            const delta = e.deltaY > 0 ? 0.8 : 1.25;
            const newScale = Math.min(Math.max(1, scale * delta), 5);
            if (newScale !== scale && newScale >= 1) {
                e.preventDefault();
                const rect = panZoomContainer.getBoundingClientRect();
                const mouseX = e.clientX - rect.left - rect.width / 2;
                const mouseY = e.clientY - rect.top - rect.height / 2;
                const scaleRatio = newScale / scale;
                pointX = mouseX - (mouseX - pointX) * scaleRatio;
                pointY = mouseY - (mouseY - pointY) * scaleRatio;
                scale = newScale;
                setTransform();
                if (slider) slider.value = scale;
                updateCursor();
            }
        }, { passive: false });
        
        // Mouse drag pan - track drag vs click
        panZoomContainer.addEventListener('mousedown', function(e) {
            if (scale > 1 && e.target === mediaElement) {
                dragStartX = e.clientX;
                dragStartY = e.clientY;
                startX = e.clientX - pointX;
                startY = e.clientY - pointY;
                panning = true;
                hasDragged = false;
                mediaElement.style.cursor = 'grabbing';
                panZoomContainer.style.cursor = 'grabbing';
            }
        });
        
        document.addEventListener('mousemove', function(e) {
            if (panning) {
                // Check if we actually moved enough to consider it a drag
                if (Math.abs(e.clientX - dragStartX) > 3 || Math.abs(e.clientY - dragStartY) > 3) {
                    hasDragged = true;
                }
                e.preventDefault();
                pointX = e.clientX - startX;
                pointY = e.clientY - startY;
                setTransform();
            }
        });
        
        document.addEventListener('mouseup', function() {
            panning = false;
            mediaElement.style.cursor = scale > 1 ? 'grab' : 'default';
            panZoomContainer.style.cursor = scale > 1 ? 'grab' : 'default';
        });
        
        function updateCursor() {
            mediaElement.style.cursor = scale > 1 ? 'grab' : 'default';
            panZoomContainer.style.cursor = scale > 1 ? 'grab' : 'default';
        }
        
        // Create zoom slider
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '1';
        slider.max = '5';
        slider.step = '0.25';
        slider.value = '1';
        slider.className = 'zoom-slider';
        slider.oninput = function() {
            const newScale = parseFloat(slider.value);
            const scaleRatio = newScale / scale;
            pointX = pointX * scaleRatio;
            pointY = pointY * scaleRatio;
            scale = newScale;
            setTransform();
            updateCursor();
        };
        zoomSlider = slider;
        
    } else {
        // For images: pan-and-zoom viewer like data/
        panZoomContainer = document.createElement('div');
        panZoomContainer.className = 'panZoomContainer';
        panZoomContainer.style.overflow = 'hidden';
        panZoomContainer.style.width = '100%';
        panZoomContainer.style.height = '100%';
        panZoomContainer.style.display = 'flex';
        panZoomContainer.style.alignItems = 'center';
        panZoomContainer.style.justifyContent = 'center';
        panZoomContainer.style.position = 'relative';
        panZoomContainer.style.cursor = 'grab';
        
        mediaElement = document.createElement('img');
        mediaElement.src = url;
        mediaElement.id = 'media-element';
        mediaElement.style.maxWidth = '100%';
        mediaElement.style.maxHeight = '100%';
        mediaElement.style.objectFit = 'contain';
        mediaElement.style.transformOrigin = 'center center';
        mediaElement.style.transition = 'transform 0.1s ease-out';
        mediaElement.style.cursor = 'grab';
        panZoomContainer.appendChild(mediaElement);
        wrapper.appendChild(panZoomContainer);
        
        // Pan and zoom state
        let scale = 1;
        let panning = false;
        let pointX = 0;
        let pointY = 0;
        let startX = 0;
        let startY = 0;
        
        function clampPosition() {
            if (scale <= 1) {
                // Snap to center when not zoomed in
                pointX = 0;
                pointY = 0;
            } else {
                // When zoomed, clamp to image borders (don't show empty space)
                const rect = panZoomContainer.getBoundingClientRect();
                const imgWidth = rect.width * scale;
                const imgHeight = rect.height * scale;
                const maxX = (imgWidth - rect.width) / 2;
                const maxY = (imgHeight - rect.height) / 2;
                pointX = Math.max(-maxX, Math.min(maxX, pointX));
                pointY = Math.max(-maxY, Math.min(maxY, pointY));
            }
        }
        
        function setTransform() {
            clampPosition();
            mediaElement.style.transform = 'translate(' + pointX + 'px, ' + pointY + 'px) scale(' + scale + ')';
        }
        
        // Get natural image dimensions for proper centering
        let imgNaturalWidth = 0;
        let imgNaturalHeight = 0;
        mediaElement.onload = function() {
            imgNaturalWidth = this.naturalWidth;
            imgNaturalHeight = this.naturalHeight;
        };
        
        // Mouse wheel zoom (follows cursor)
        panZoomContainer.addEventListener('wheel', function(e) {
            const delta = e.deltaY > 0 ? 0.8 : 1.25; // zoom out/in multipliers
            const newScale = Math.min(Math.max(1, scale * delta), 5);
            if (newScale !== scale && newScale >= 1) {
                e.preventDefault();
                // Get mouse position relative to container center
                const rect = panZoomContainer.getBoundingClientRect();
                const mouseX = e.clientX - rect.left - rect.width / 2;
                const mouseY = e.clientY - rect.top - rect.height / 2;
                // Calculate where the mouse point will be after zoom
                const scaleRatio = newScale / scale;
                // Adjust pan so mouse point stays at same screen position
                pointX = mouseX - (mouseX - pointX) * scaleRatio;
                pointY = mouseY - (mouseY - pointY) * scaleRatio;
                scale = newScale;
                setTransform();
                if (slider) slider.value = scale;
            }
            updateCursor();
        }, { passive: false });
        
        // Mouse drag pan
        panZoomContainer.addEventListener('mousedown', function(e) {
            if (scale > 1) {
                e.preventDefault();
                startX = e.clientX - pointX;
                startY = e.clientY - pointY;
                panning = true;
                mediaElement.style.cursor = 'grabbing';
                panZoomContainer.style.cursor = 'grabbing';
            }
        });
        
        document.addEventListener('mousemove', function(e) {
            if (panning) {
                e.preventDefault();
                pointX = e.clientX - startX;
                pointY = e.clientY - startY;
                setTransform();
            }
        });
        
        document.addEventListener('mouseup', function() {
            panning = false;
            mediaElement.style.cursor = scale > 1 ? 'grab' : 'default';
            panZoomContainer.style.cursor = scale > 1 ? 'grab' : 'default';
        });
        
        function updateCursor() {
            mediaElement.style.cursor = scale > 1 ? 'grab' : 'default';
            panZoomContainer.style.cursor = scale > 1 ? 'grab' : 'default';
        }
        
        // Create zoom slider (reuse the outer variable)
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '1';
        slider.max = '5';
        slider.step = '0.25';
        slider.value = '1';
        slider.className = 'zoom-slider';
        slider.oninput = function() {
            const newScale = parseFloat(slider.value);
            // Keep the center point stationary during slider zoom
            const scaleRatio = newScale / scale;
            pointX = pointX * scaleRatio;
            pointY = pointY * scaleRatio;
            scale = newScale;
            setTransform();
            updateCursor();
        };
        // Assign to outer scope variable for controls section
        zoomSlider = slider;
    }
    
    // Controls
    const controls = document.createElement('div');
    controls.className = 'media-controls';
    
    // Version buttons if variants available
    if (variants && variants.length > 1) {
        const toggleGroup = document.createElement('div');
        toggleGroup.className = 'toggle-group';
        
        const toggleLabel = document.createElement('span');
        toggleLabel.className = 'toggle-label';
        toggleLabel.textContent = <?php echo json_encode(($t['player_version'] ?? 'Version') . ':'); ?>;
        toggleGroup.appendChild(toggleLabel);
        
        // Build buttons for other versions only, rebuild on switch
        let currentFile = url.split('/').pop();
        function updateVersionButtons() {
            toggleGroup.querySelectorAll('.media-btn').forEach(b => b.remove());
            variants.forEach((variant) => {
                const variantFile = variant.url.split('/').pop();
                if (variantFile === currentFile) return; // Skip current
                const btn = document.createElement('button');
                btn.className = 'media-btn';
                btn.textContent = variant.label;
                btn.onclick = function() {
                    // Switch media
                    if (type === 'video') {
                        const currentTime = mediaElement.currentTime;
                        mediaElement.src = variant.url;
                        currentFile = variantFile;
                        mediaElement.onloadedmetadata = function() {
                            mediaElement.currentTime = currentTime;
                        };
                    } else {
                        mediaElement.src = variant.url;
                        currentFile = variantFile;
                    }
                    // Update title with new version and rebuild buttons
                    const titleEl = header.querySelector('h3');
                    if (titleEl) titleEl.textContent = (stationLabel || '') + ' - ' + (variant.desc || variant.label);
                    updateVersionButtons();
                };
                toggleGroup.appendChild(btn);
            });
        }
        updateVersionButtons();
        controls.appendChild(toggleGroup);
    }
    
    // Zoom controls for images
    if (zoomSlider) {
        const zoomLabel = document.createElement('span');
        zoomLabel.className = 'zoom-label';
        zoomLabel.textContent = <?php echo json_encode(($t['player_zoom'] ?? 'Zoom') . ':'); ?>;
        zoomLabel.style.color = '#c01010';
        zoomLabel.style.fontWeight = 'bold';
        const zoomWrap = document.createElement('div');
        zoomWrap.style.display = 'flex';
        zoomWrap.style.alignItems = 'center';
        zoomWrap.style.whiteSpace = 'nowrap';
        zoomWrap.style.zIndex = '10';
        zoomWrap.appendChild(zoomLabel);
        zoomWrap.appendChild(zoomSlider);
        controls.appendChild(zoomWrap);
    }
    
    // Download button
    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'media-btn';
    downloadBtn.textContent = <?php echo json_encode($t['player_download'] ?? 'Download'); ?>;
    downloadBtn.onclick = function() {
        const a = document.createElement('a');
        a.href = document.getElementById('media-element').src || url;
        a.download = a.href.split('/').pop();
        a.click();
    };
    // Fullscreen button - only media, no controls
    const fullscreenBtn = document.createElement('button');
    fullscreenBtn.className = 'media-btn';
    fullscreenBtn.textContent = <?php echo json_encode($t['player_fullscreen'] ?? 'Fullscreen'); ?>;
    fullscreenBtn.onclick = function() {
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            // Request fullscreen on the pan/zoom container to keep zoom/pan working
            panZoomContainer.requestFullscreen();
        }
    };
    // Reset zoom when entering fullscreen so video fills screen
    panZoomContainer.addEventListener('fullscreenchange', function() {
        if (document.fullscreenElement) {
            // Reset zoom to fill fullscreen
            scale = 1;
            pointX = 0;
            pointY = 0;
            setTransform();
            if (zoomSlider) zoomSlider.value = 1;
            updateCursor();
        }
    });
    // Frame step and play/pause buttons for video - wrapped to prevent linebreak
    if (type === 'video') {
        const frameControls = document.createElement('span');
        frameControls.style.whiteSpace = 'nowrap';
        
        const frameBackBtn = document.createElement('button');
        frameBackBtn.className = 'frame-btn';
        frameBackBtn.textContent = '◀';
        frameBackBtn.title = <?php echo json_encode($t['player_frame_back'] ?? 'Step back 1 frame'); ?>;
        frameBackBtn.onclick = function() {
            if (mediaElement.readyState >= 2) {
                const fps = 25; // Assume 25fps
                mediaElement.currentTime = Math.max(0, mediaElement.currentTime - (1/fps));
            }
        };
        frameControls.appendChild(frameBackBtn);
        
        const playPauseBtn = document.createElement('button');
        playPauseBtn.className = 'frame-btn';
        playPauseBtn.textContent = '⏸';
        playPauseBtn.title = <?php echo json_encode($t['player_pause'] ?? 'Pause'); ?>;
        playPauseBtn.onclick = function() {
            if (mediaElement.paused) {
                mediaElement.play();
                playPauseBtn.textContent = '⏸';
                playPauseBtn.title = <?php echo json_encode($t['player_pause'] ?? 'Pause'); ?>;
            } else {
                mediaElement.pause();
                playPauseBtn.textContent = '▶';
                playPauseBtn.title = <?php echo json_encode($t['player_play'] ?? 'Play'); ?>;
            }
        };
        // Update button when video state changes
        mediaElement.onplay = function() {
            playPauseBtn.textContent = '⏸';
            playPauseBtn.title = <?php echo json_encode($t['player_pause'] ?? 'Pause'); ?>;
        };
        mediaElement.onpause = function() {
            playPauseBtn.textContent = '▶';
            playPauseBtn.title = <?php echo json_encode($t['player_play'] ?? 'Play'); ?>;
        };
        frameControls.appendChild(playPauseBtn);
        
        const frameForwardBtn = document.createElement('button');
        frameForwardBtn.className = 'frame-btn';
        frameForwardBtn.textContent = '▶';
        frameForwardBtn.title = <?php echo json_encode($t['player_frame_forward'] ?? 'Step forward 1 frame'); ?>;
        frameForwardBtn.onclick = function() {
            if (mediaElement.readyState >= 2) {
                const fps = 25; // Assume 25fps
                mediaElement.currentTime = Math.min(mediaElement.duration, mediaElement.currentTime + (1/fps));
            }
        };
        frameControls.appendChild(frameForwardBtn);
        
        controls.appendChild(frameControls);
    }
    
    controls.appendChild(fullscreenBtn);
    
    controls.appendChild(downloadBtn);
    
    // Brightness and Contrast sliders for video and images
    if (type === 'video' || type === 'image') {
        const filterControls = document.createElement('div');
        filterControls.style.display = 'flex';
        filterControls.style.gap = '10px';
        filterControls.style.alignItems = 'center';
        filterControls.style.marginLeft = 'auto';
        
        // Brightness slider
        const brightnessWrap = document.createElement('div');
        brightnessWrap.style.display = 'flex';
        brightnessWrap.style.flexDirection = 'column';
        brightnessWrap.style.alignItems = 'center';
        const brightnessLabel = document.createElement('label');
        brightnessLabel.textContent = <?php echo json_encode($t['player_brightness'] ?? 'Brightness'); ?>;
        brightnessLabel.style.fontSize = '11px';
        brightnessLabel.style.color = '#8aa4be';
        const brightnessSlider = document.createElement('input');
        brightnessSlider.type = 'range';
        brightnessSlider.min = '0.5';
        brightnessSlider.max = '2';
        brightnessSlider.step = '0.1';
        brightnessSlider.value = '1';
        brightnessSlider.style.width = '80px';
        brightnessSlider.style.accentColor = '#4a9fd4';
        brightnessWrap.appendChild(brightnessLabel);
        brightnessWrap.appendChild(brightnessSlider);
        
        // Contrast slider
        const contrastWrap = document.createElement('div');
        contrastWrap.style.display = 'flex';
        contrastWrap.style.flexDirection = 'column';
        contrastWrap.style.alignItems = 'center';
        const contrastLabel = document.createElement('label');
        contrastLabel.textContent = <?php echo json_encode($t['player_contrast'] ?? 'Contrast'); ?>;
        contrastLabel.style.fontSize = '11px';
        contrastLabel.style.color = '#8aa4be';
        const contrastSlider = document.createElement('input');
        contrastSlider.type = 'range';
        contrastSlider.min = '0.5';
        contrastSlider.max = '2';
        contrastSlider.step = '0.1';
        contrastSlider.value = '1';
        contrastSlider.style.width = '80px';
        contrastSlider.style.accentColor = '#4a9fd4';
        contrastWrap.appendChild(contrastLabel);
        contrastWrap.appendChild(contrastSlider);
        
        // Reset button
        const resetFiltersBtn = document.createElement('button');
        resetFiltersBtn.className = 'media-btn';
        resetFiltersBtn.textContent = <?php echo json_encode($t['player_reset_filters'] ?? 'Reset'); ?>;
        resetFiltersBtn.style.fontSize = '11px';
        resetFiltersBtn.style.padding = '4px 8px';
        
        // Update filters function
        function updateFilters() {
            const brightness = brightnessSlider.value;
            const contrast = contrastSlider.value;
            mediaElement.style.filter = 'brightness(' + brightness + ') contrast(' + contrast + ')';
        }
        
        brightnessSlider.addEventListener('input', updateFilters);
        contrastSlider.addEventListener('input', updateFilters);
        
        resetFiltersBtn.onclick = function() {
            brightnessSlider.value = 1;
            contrastSlider.value = 1;
            updateFilters();
        };
        
        filterControls.appendChild(resetFiltersBtn);
        filterControls.appendChild(brightnessWrap);
        filterControls.appendChild(contrastWrap);
        controls.appendChild(filterControls);
    }
    
    modal.appendChild(header);
    modal.appendChild(wrapper);
    modal.appendChild(controls);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);
    document.body.style.overflow = 'hidden';
}

function closeMediaPlayer() {
    const modal = document.getElementById('media-modal');
    if (modal) {
        modal.remove();
        document.body.style.overflow = '';
    }
}

// Close on Escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeMediaPlayer();
        closeTextViewer();
    }
});

// Text file viewer
function openTextViewer(url, title) {
    const existing = document.getElementById('text-modal');
    if (existing) existing.remove();
    
    const backdrop = document.createElement('div');
    backdrop.id = 'text-modal';
    backdrop.className = 'modal-backdrop';
    backdrop.onclick = function(e) {
        if (e.target === backdrop) closeTextViewer();
    };
    
    const modal = document.createElement('div');
    modal.className = 'modal-content';
    modal.style.minWidth = '600px';
    
    // Header
    const header = document.createElement('div');
    header.className = 'modal-header';
    const titleEl = document.createElement('h3');
    titleEl.textContent = title;
    const closeBtn = document.createElement('button');
    closeBtn.className = 'modal-close';
    closeBtn.innerHTML = '&times;';
    closeBtn.onclick = closeTextViewer;
    header.appendChild(titleEl);
    header.appendChild(closeBtn);
    
    // Content area
    const content = document.createElement('div');
    content.className = 'text-viewer-content';
    content.innerHTML = '<div style="text-align:center;padding:20px;">Loading...</div>';
    
    // Controls
    const controls = document.createElement('div');
    controls.className = 'media-controls';
    
    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'media-btn';
    downloadBtn.textContent = <?php echo json_encode($t['player_download'] ?? 'Download'); ?>;
    downloadBtn.onclick = function() {
        const a = document.createElement('a');
        a.href = url;
        a.download = url.split('/').pop();
        a.click();
    };
    controls.appendChild(downloadBtn);
    
    modal.appendChild(header);
    modal.appendChild(content);
    modal.appendChild(controls);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);
    document.body.style.overflow = 'hidden';
    
    // Load and format content
    fetch(url)
        .then(r => r.text())
        .then(text => {
            content.innerHTML = formatTextContent(text, url);
        })
        .catch(err => {
            content.innerHTML = '<div style="color:#f44;">Error loading file: ' + err.message + '</div>';
        });
}

function formatTextContent(text, url) {
    const lines = text.split('\n');
    const filename = url.split('/').pop();
    
    // Check file type
    if (filename.includes('centroid') || filename.includes('light')) {
        // Data table format
        return formatDataTable(lines);
    } else if (text.includes('[') && text.includes(']')) {
        // INI format
        return formatIniFile(lines);
    } else {
        // Plain text
        return '<pre class="text-viewer-pre">' + escapeHtml(text) + '</pre>';
    }
}

function formatDataTable(lines) {
    let html = '<table style="width:100%;border-collapse:collapse;">';
    lines.forEach((line, idx) => {
        if (!line.trim()) return;
        const cells = line.trim().split(/\s+/);
        html += '<tr class="text-row">';
        cells.forEach(cell => {
            const isNumber = /^[\d.-]+$/.test(cell);
            const isDate = /\d{4}-\d{2}-\d{2}/.test(cell);
            const cssClass = isNumber ? 'text-number' : (isDate ? 'text-timestamp' : '');
            html += '<td style="padding:3px 8px;"><span class="' + cssClass + '">' + escapeHtml(cell) + '</span></td>';
        });
        html += '</tr>';
    });
    html += '</table>';
    return html;
}

function formatIniFile(lines) {
    let html = '<pre class="text-viewer-pre">';
    lines.forEach(line => {
        if (line.match(/^\[.+\]$/)) {
            // Section header
            html += '<span class="text-section">' + escapeHtml(line) + '</span>\n';
        } else if (line.match(/^\s*#/)) {
            // Comment
            html += '<span class="text-comment">' + escapeHtml(line) + '</span>\n';
        } else if (line.includes('=')) {
            // Key=value
            const [key, ...valParts] = line.split('=');
            const value = valParts.join('=');
            html += '<span class="text-key">' + escapeHtml(key) + '</span>=<span class="text-value">' + escapeHtml(value) + '</span>\n';
        } else {
            html += escapeHtml(line) + '\n';
        }
    });
    html += '</pre>';
    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function closeTextViewer() {
    const modal = document.getElementById('text-modal');
    if (modal) {
        modal.remove();
        document.body.style.overflow = '';
    }
}
</script>

</body>
</html>
