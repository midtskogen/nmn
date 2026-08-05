<!doctype html public "-//W3C//DTD HTML 4.0 Transitional//EN">
<html>
<head>

<style type="text/css">
h1   {text-align: center; color:#602008}
h2   {text-align: center}
body {
  background: #ffffff;
  color: #000018;
  font-family: Verdana, Geneva, Arial, Helvetica, sans-serif;
  font-size : 12px
}
table {
  color: #000018;
  font-family: Verdana, Geneva, Arial, Helvetica, sans-serif;
  font-size : 12px
}
.submit input
{
    border: 1;
    background-color: #ffffff;
    color: #c01010;
    text-decoration: underline;
    font-family: Verdana, Geneva, Arial, Helvetica, sans-serif;
    font-size: 12px
    overflow: visible;
}

form,
.submit,
.submit input
{
    display: inline;
}

a:link {color:#c01010}
a:visited {color:#601010}
</style>

<title>Observasjoner</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">

<?php
if ($_POST) {
  foreach ($_POST as $c) {print($c . " "); }
}
?>

</body>
</html>
