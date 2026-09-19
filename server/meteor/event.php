<html>
<body>

<?php
if (!isset($_GET["time"])) {
    http_response_code(400);
    exit;
}
$time = rtrim(preg_replace("([^\w\s\d\./\-_~,:\[\]\(\]]|[\.]{2,})", '', $_GET["time"]), '/');
$date = DateTime::createFromFormat('Ymd/His', $time);
if ($date === false) {
    http_response_code(400);
    exit;
}

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
