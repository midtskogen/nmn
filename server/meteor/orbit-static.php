<?php

// Cron/CLI generator only: a web hit must not trigger a full directory
// rescan and a file write on every request.
if (php_sapi_name() !== 'cli') {
    http_response_code(403);
    exit('Forbidden');
}

ob_start();
chdir("/home/httpd/norskmeteornettverk.no/meteor");
$myDirectory = opendir(".");
while($entryName = readdir($myDirectory)) { $dirArray[] = $entryName; }
closedir($myDirectory);
$indexCount = count($dirArray);
rsort($dirArray);

$dates = array();

$filtered = array();
for ($index=0; $index < $indexCount; $index++) {
	if (substr("$dirArray[$index]", 0, 1) != "." && substr("$dirArray[$index]", 0, 4) == date('Y') && filetype($dirArray[$index]) == 'dir') {
		$myDirectory2 = opendir($dirArray[$index]);
		while($entryName2 = readdir($myDirectory2)) { $dirArray2[] = $entryName2; }
		closedir($myDirectory2);
		$indexCount2 = count($dirArray2);
		rsort($dirArray2);
		for ($index2=0; $index2 < $indexCount2; $index2++) {
	        	if (preg_match("/^\d{6}(_\d+)?$/", $dirArray2[$index2]) && file_exists($dirArray[$index] . "/" . $dirArray2[$index2] . "/index.php")) {
			        $year = substr($dirArray[$index], 0, 4);
				$month = substr($dirArray[$index], 4, 2);
				$dates[$year][$month][] = $dirArray[$index] . "/" . $dirArray2[$index2];
				$filtered[] = "$dirArray[$index]/$dirArray2[$index2]" . "/";
			}
		}
		unset($dirArray2);
	}
}

$head = $filtered[0];

foreach ($dates as $year) {
  reset($year);
  $firstKey = key($year);
  $firstValue = current($year);
  $y = substr($firstValue[0], 0, 4);
  print("<table width=\"100%\"><tr><th align=center>$y<br>\n");
  print("<table border=1><tr>\n");
  foreach ($year as $month) {
    print("<td valign=top align=center>\n");
    $m = substr($month[0], 4, 2);
    if ($m == "01") { print("<b>januar</b><hr>"); }
    if ($m == "02") { print("<b>februar</b><hr>"); }
    if ($m == "03") { print("<b>mars</b><hr>"); }
    if ($m == "04") { print("<b>april</b><hr>"); }
    if ($m == "05") { print("<b>mai</b><hr>"); }
    if ($m == "06") { print("<b>juni</b><hr>"); }
    if ($m == "07") { print("<b>juli</b><hr>"); }
    if ($m == "08") { print("<b>august</b><hr>"); }
    if ($m == "09") { print("<b>september</b><hr>"); }
    if ($m == "10") { print("<b>oktober</b><hr>"); }
    if ($m == "11") { print("<b>november</b><hr>"); }
    if ($m == "12") { print("<b>desember</b><hr>"); }
    foreach ($month as $date) {
      if (!file_exists($date . "/orbit.jpg")) { continue; }
      $y = substr($date, 0, 4);
      $m = substr($date, 4, 2);
      $d = substr($date, 6, 2);
      $d1 = substr($date, 6, 2);
      $d2 = substr($date, 9, 6);
      $time = substr_replace($d1 . ". " . $d2, ":", 6, 0);
      $time = substr_replace($time, ":", 9, 0);
      $d2 = substr_replace($d2, ":", 2, 0);
      $d2 = substr_replace($d2, ":", 5, 0);
      if (file_exists($date . "/map.jpg")) {
        $res_raw = @file_get_contents($date . "/obs_" . $y . "-" . $m . "-" . $d . "_" . $d2 . ".res");
        $res = $res_raw !== false ? preg_split("/[\s]+/", $res_raw) : [];
        $start = $res[5] ?? 0;
        $end = $res[11] ?? 0;
        if ($start > 150 || $end < 10 || $end > 150 || $start < 10) {
          $b1 = "<font color=lightgray>";
	} else if ($start > 60 && $end < 40) {
          $b1 = "<font color=red>";
        } else {
          $b1 = "<font color=black>";
        }

	$dirs = array_map('basename', array_filter(glob($date . '/*'), 'is_dir'));
        $loc = "";
        if (file_exists($date . "/location.txt")) {
          $loc = "<br><small>" . htmlspecialchars(trim((string)@file_get_contents($date . "/location.txt"))) . "</small>";
        } else {
	  $loc = "<br><small>(" . htmlspecialchars(implode(", ", $dirs)) . ")</small>";
	}
        $b2 = "$loc</font>";
      } else {
        $dirs = array_map('basename', array_filter(glob($date . '/*'), 'is_dir'));
        $loc = "<br><small>(" . htmlspecialchars(implode(", ", $dirs)) . ")</small>";
        $b1 = "<font color=grey>";
        $b2 = "$loc</font>";
      }
      $dirs2 = [];
      print(htmlspecialchars($date) . '<br>');
      foreach($dirs as $d) {
	$cams = array_map('basename', array_filter(glob($date . '/' . $d . '/cam*'), 'is_dir'));
	foreach($cams as $c) { $dirs2[] = $date . '/' . $d . '/' . $c;  print(htmlspecialchars($d . '/' . $c) . "<br>"); }
      }
      foreach($dirs2 as $d) { $f = $d . '/fireball.jpg'; if (file_exists($f)) { print("<img src=" . htmlspecialchars($f, ENT_QUOTES) . " width=256><br>"); } }
      print("<a href=\"" . htmlspecialchars($date, ENT_QUOTES) . "/\">$b1$time$b2</a><br><hr>");
    }
    print("</td>\n");
  }
  print("</tr></table>\n");
  print("</th></tr></table>\n");
}

// Atomic write so web readers never see a partial file.
$_orb_tmp = '/home/httpd/norskmeteornettverk.no/meteor/orbit-static.html.tmp.' . getmypid();
file_put_contents($_orb_tmp, ob_get_clean());
rename($_orb_tmp, '/home/httpd/norskmeteornettverk.no/meteor/orbit-static.html');
					 
?>
