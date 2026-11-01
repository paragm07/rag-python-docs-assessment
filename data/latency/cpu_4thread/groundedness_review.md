# Groundedness review (manual)

For each answered question, check every factual claim against the cited passage.
Mark each claim supported / not supported, and record the result in the report.

## q01 (single_passage)
Question: What does json.dumps do with the sort_keys argument set to True?

Answer: When the `sort_keys` argument is set to True, `json.dumps` will sort the keys of dictionaries in the output JSON, making the JSON output consistent and easier to compare on a day-to-day basis.

Citations:
- python-3.12.7/json#0002 — python-3.12.7/json.rst:7-131 (:mod:`!json` --- JSON encoder and decoder)

Supported: [ ] all claims  [ ] partly  [ ] no

## q02 (single_passage)
Question: In what unit is the timeout argument of urllib.request.urlopen measured?

Answer: seconds

Citations:
- python-3.12.7/urllib.request#0060 — python-3.12.7/urllib.request.rst:658-669 (:mod:`!urllib.request` --- Extensible library for opening URLs > OpenerDirector Objects)
- python-3.12.7/urllib.request#0002 — python-3.12.7/urllib.request.rst:37-131 (:mod:`!urllib.request` --- Extensible library for opening URLs)

Supported: [ ] all claims  [ ] partly  [ ] no

## q03 (single_passage)
Question: What is the connect timeout in requests?

Answer: The connect timeout is the number of seconds Requests will wait for your client to establish a connection to a remote machine (corresponding to the `connect()` call on the socket).

Citations:
- requests-2.32.3/advanced#0047 — requests-2.32.3/advanced.rst:1093-1141 (Advanced Usage > Timeouts)

Supported: [ ] all claims  [ ] partly  [ ] no

## q04 (single_passage)
Question: How do I enable automatic retries for failed connections in requests?

Answer: To enable automatic retries for failed connections in Requests, you can use the `urllib3.util.Retry` class within a `requests.Session` object, as shown in the example provided in the documentation.

Citations:
- requests-2.32.3/advanced#0044 — requests-2.32.3/advanced.rst:1038-1060 (Advanced Usage > Transport Adapters > Example: Automatic Retries)

Supported: [ ] all claims  [ ] partly  [ ] no

## q05 (single_passage)
Question: Which exception does r.json() raise when the body is not valid JSON in requests?

Answer: ``r.json()`` raises ``requests.exceptions.JSONDecodeError`` when the body is not valid JSON.

Citations:
- requests-2.32.3/quickstart#0008 — requests-2.32.3/quickstart.rst:146-164 (Quickstart > JSON Response Content)

Supported: [ ] all claims  [ ] partly  [ ] no

## q07 (single_passage)
Question: What does string.Template.safe_substitute do when a placeholder has no value in the mapping?

Answer: When a placeholder has no value in the mapping, instead of raising a KeyError exception, the original placeholder will appear in the resulting string intact.

Citations:
- python-3.12.7/string#0060 — python-3.12.7/string.rst:786-801 (:mod:`!string` --- Common string operations > Template strings)

Supported: [ ] all claims  [ ] partly  [ ] no

## q09 (single_passage)
Question: Which HTTP version does urllib.request use, and what connection header does it send?

Answer: urllib.request module uses HTTP/1.1 and includes `Connection:close` header in its HTTP requests.

Citations:
- python-3.12.7/urllib.request#0002 — python-3.12.7/urllib.request.rst:37-131 (:mod:`!urllib.request` --- Extensible library for opening URLs)

Supported: [ ] all claims  [ ] partly  [ ] no

## q11 (table_lookup)
Question: What range of values does the %U directive produce in datetime strftime?

Answer: %U directive produces week number of the year (Sunday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Sunday are considered to be in week 0.

Citations:
- python-3.12.7/datetime#0196 — python-3.12.7/datetime.rst:2465-2472 (:mod:`!datetime` --- Basic date and time types > :meth:`~.datetime.strftime` and :meth:`~.datetime.strptime` Behavior > :meth:`~.datetime.strftime` and :meth:`~.datetime.strptime` Format Codes)
- python-3.12.7/datetime#0192 — python-3.12.7/datetime.rst:2428-2435 (:mod:`!datetime` --- Basic date and time types > :meth:`~.datetime.strftime` and :meth:`~.datetime.strptime` Behavior > :meth:`~.datetime.strftime` and :meth:`~.datetime.strptime` Format Codes)

Supported: [ ] all claims  [ ] partly  [ ] no

## q12 (table_lookup)
Question: Which Python type does a JSON array decode to with json.loads?

Answer: list

Citations:
- python-3.12.7/json#0011 — python-3.12.7/json.rst:305-374 (:mod:`!json` --- JSON encoder and decoder > Encoders and Decoders)

Supported: [ ] all claims  [ ] partly  [ ] no

## q13 (table_lookup)
Question: What strings does the %p directive produce in the en_US locale?

Answer: The %p directive produces 'AM' or 'PM' in the en_US locale.

Citations:
- python-3.12.7/datetime#0188 — python-3.12.7/datetime.rst:2398-2406 (:mod:`!datetime` --- Basic date and time types > :meth:`~.datetime.strftime` and :meth:`~.datetime.strptime` Behavior > :meth:`~.datetime.strftime` and :meth:`~.datetime.strptime` Format Codes)

Supported: [ ] all claims  [ ] partly  [ ] no

## q14 (multi_document)
Question: Are the timeout limits in requests and urllib.request.urlopen both measured in seconds and both applied to the connection attempt?

Answer: The timeout limits in requests are measured in seconds and applied to the connection attempt. In urllib.request.urlopen, the timeout is also measured in seconds and applied to the connection attempt, but it only works for HTTP, HTTPS, and FTP connections.

Citations:
- python-3.12.7/urllib.request#0002 — python-3.12.7/urllib.request.rst:37-131 (:mod:`!urllib.request` --- Extensible library for opening URLs)
- requests-2.32.3/advanced#0048 — requests-2.32.3/advanced.rst:1093-1141 (Advanced Usage > Timeouts)

Supported: [ ] all claims  [ ] partly  [ ] no

## q15 (multi_document)
Question: Which exception does json.loads raise for invalid JSON, and which one does requests raise for r.json()?

Answer: requests.exceptions.JSONDecodeError for r.json() and json.JSONDecodeError for json.loads

Citations:
- python-3.12.7/json#0032 — python-3.12.7/json.rst:619-628 (:mod:`!json` --- JSON encoder and decoder > Standard Compliance and Interoperability > Repeated Names Within an Object)

Supported: [ ] all claims  [ ] partly  [ ] no
