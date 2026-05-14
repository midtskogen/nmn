<?php

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
	if (substr("$dirArray[$index]", 0, 1) != "." && substr("$dirArray[$index]", 0, 4) == "2025" && filetype($dirArray[$index]) == 'dir') {
		$myDirectory2 = opendir($dirArray[$index]);
		while($entryName2 = readdir($myDirectory2)) { $dirArray2[] = $entryName2; }
		closedir($myDirectory2);
		$indexCount2 = count($dirArray2);
		rsort($dirArray2);
		for ($index2=0; $index2 < $indexCount2; $index2++) {
	        	if (preg_match("/\d{6}/", $dirArray2[$index2]) && substr("$dirArray2[$index2]", 0, 1) != "." && file_exists($dirArray[$index] . "/" . $dirArray2[$index2] . "/index.php")) {
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

print("<form action=\"id_check.php\" method=\"post\">");
print("<input type=\"submit\" value=\"Se avkrysninger\">");

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
        $res = shell_exec("cat " . $date . "/obs_" . $y . "-" . $m . "-" . $d . "_" . $d2 . ".res");
        $res = preg_split("/[\s]+/", $res);
        $start = $res[5];
        $end = $res[11];
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
          $loc = "<br><small>" . trim(shell_exec("cat " . $date . "/location.txt")) . "</small>";
        } else {
	  $loc = "<br><small>(" . implode(", ", $dirs) . ")</small>";
	}
        $b2 = "$loc</font>";
      } else {
        $dirs = array_map('basename', array_filter(glob($date . '/*'), 'is_dir'));
        $loc = "<br><small>(" . implode(", ", $dirs) . ")</small>";
        $b1 = "<font color=grey>";
        $b2 = "$loc</font>";
      }
      $dirs2 = [];
      print("<input type=\"checkbox\" id=\"wrong\" name=\"$date\" value=\"$date\">");
      print($date . '<br>');
      foreach($dirs as $d) {
	$cams = array_map('basename', array_filter(glob($date . '/' . $d . '/cam*'), 'is_dir'));
	foreach($cams as $c) { $dirs2[] = $date . '/' . $d . '/' . $c;  print($d . '/' . $c . "<br>"); }
      }
      foreach($dirs2 as $d) {
        $fireball_path = $d . '/fireball.jpg';
        $fireball_orig_path = $d . '/fireball_orig.jpg';
        $image_to_display = '';

        if (file_exists($fireball_path)) {
          $image_to_display = $fireball_path;
        } else if (file_exists($fireball_orig_path)) {
          $image_to_display = $fireball_orig_path;
        }

        if (!empty($image_to_display)) {
          print("<input type=\"checkbox\" name=\"fireball_dirs[]\" value=\"$d\">");
          print("<img src=\"$image_to_display\" width=256><br>");
        }
      }
      print("<a href=\"$date" . "/\">$b1$time$b2</a><br><hr>");
    }
    print("</td>\n");
  }
  print("</tr></table>\n");
  print("</th></tr></table>\n");
}

file_put_contents('/home/httpd/norskmeteornettverk.no/meteor/id-static.html', ob_get_clean());
					 
?>
