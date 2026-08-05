<html>
<body>

<?php
$time = rtrim(preg_replace("([^\w\s\d\./\-_~,:\[\]\(\]]|[\.]{2,})", '', $_GET["time"]), '/');
$date = DateTime::createFromFormat('Ymd/His', $time);

$plus = clone $date;
$minus = clone $date;
$found = 0;
for ($i = 0; $i < 7; $i++) {
  $found |= file_exists($plus->format('Ymd/His')) | file_exists($minus->format('Ymd/His')) ;
  $minus->sub(new DateInterval('PT1S'));
  $plus->add(new DateInterval('PT1S'));
}

echo $found;

?>

</body>
</html>
