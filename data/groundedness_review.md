# Groundedness review (manual)

Draft verdicts written during the debugging pass; each one must be checked by the author against
the cited passage before submission. Summary: 8 supported, 2 partly (claim not in the cited passage),
2 supported but incomplete, 1 partly unsupported, 1 wrong.

For each answered question, check every factual claim against the cited passage.
Mark each claim supported / not supported, and record the result in the report.

## q01 (single_passage)
Question: What does json.dumps do with the sort_keys argument set to True?

Answer: When the sort_keys argument is set to True, json.dumps will sort the keys of dictionaries in the output JSON, making the JSON output consistent and useful for regression tests.

Citations:
- python-3.12.7/json#0002 — python-3.12.7/json.rst:7-131 (:mod:`!json` --- JSON encoder and decoder)

Verdict: PARTLY - sorting keys is supported by the cited chunk; "useful for regression tests" is true (json.rst:447) but not in the cited passage.

## q02 (single_passage)
Question: In what unit is the timeout argument of urllib.request.urlopen measured?

Answer: seconds

Citations:
- python-3.12.7/urllib.request#0060 — python-3.12.7/urllib.request.rst:658-669 (:mod:`!urllib.request` --- Extensible library for opening URLs > OpenerDirector Objects)
- python-3.12.7/urllib.request#0002 — python-3.12.7/urllib.request.rst:37-131 (:mod:`!urllib.request` --- Extensible library for opening URLs)

Verdict: SUPPORTED - "seconds" is stated in the cited urlopen passage.

## q03 (single_passage)
Question: What is the connect timeout in requests?

Answer: The **connect** timeout is the number of seconds Requests will wait for your client to establish a connection to a remote machine (corresponding to the `connect()`_) call on the socket.

Citations:
- requests-2.32.3/advanced#0047 — requests-2.32.3/advanced.rst:1093-1141 (Advanced Usage > Timeouts)

Verdict: SUPPORTED - near-verbatim from advanced.rst Timeouts.

## q04 (single_passage)
Question: How do I enable automatic retries for failed connections in requests?

Answer: To enable automatic retries for failed connections in Requests, you can use the `urllib3.util.Retry` class within a `requests.Session` object, as shown in the example provided in the documentation.

Citations:
- requests-2.32.3/advanced#0044 — requests-2.32.3/advanced.rst:1038-1060 (Advanced Usage > Transport Adapters > Example: Automatic Retries)

Verdict: SUPPORTED, INCOMPLETE - Retry and Session are in the cited passage; omits mounting an HTTPAdapter with max_retries.

## q05 (single_passage)
Question: Which exception does r.json() raise when the body is not valid JSON in requests?

Answer: requests.exceptions.JSONDecodeError

Citations:
- requests-2.32.3/quickstart#0008 — requests-2.32.3/quickstart.rst:146-164 (Quickstart > JSON Response Content)

Verdict: SUPPORTED - quickstart.rst JSON Response Content.

## q07 (single_passage)
Question: What does string.Template.safe_substitute do when a placeholder has no value in the mapping?

Answer: When a placeholder has no value in the mapping, the original placeholder will appear in the resulting string intact instead of raising a KeyError exception.

Citations:
- python-3.12.7/string#0060 — python-3.12.7/string.rst:786-801 (:mod:`!string` --- Common string operations > Template strings)

Verdict: SUPPORTED - string.rst Template strings.

## q09 (single_passage)
Question: Which HTTP version does urllib.request use, and what connection header does it send?

Answer: urllib.request module uses HTTP/1.1 and includes ``Connection:close`` header in its HTTP requests.

Citations:
- python-3.12.7/urllib.request#0002 — python-3.12.7/urllib.request.rst:37-131 (:mod:`!urllib.request` --- Extensible library for opening URLs)

Verdict: SUPPORTED - urllib.request.rst, HTTP/1.1 and Connection: close.

## q10 (table_lookup)
Question: What range of values does the %j directive produce in datetime strftime?

Answer: The %j directive produces a zero-padded decimal number representing the day of the year, ranging from 001 to 366.

Citations:
- python-3.12.7/datetime#0191 — python-3.12.7/datetime.rst:2425-2427 (:mod:`!datetime` --- Basic date and time types > :meth:`~.datetime.strftime` and :meth:`~.datetime.strptime` Behavior > :meth:`~.datetime.strftime` and :meth:`~.datetime.strptime` Format Codes)

Verdict: SUPPORTED - cites the %j table row, datetime.rst:2425-2427.

## q11 (table_lookup)
Question: What range of values does the %U directive produce in datetime strftime?

Answer: The %U directive produces week numbers of the year, starting from 00 for weeks that have not yet started (preceding the first Sunday of the year), and ending with 53.

Citations:
- python-3.12.7/datetime#0192 — python-3.12.7/datetime.rst:2428-2435 (:mod:`!datetime` --- Basic date and time types > :meth:`~.datetime.strftime` and :meth:`~.datetime.strptime` Behavior > :meth:`~.datetime.strftime` and :meth:`~.datetime.strptime` Format Codes)

Verdict: SUPPORTED - cites the %U table row.

## q12 (table_lookup)
Question: Which Python type does a JSON array decode to with json.loads?

Answer: list

Citations:
- python-3.12.7/json#0011 — python-3.12.7/json.rst:305-374 (:mod:`!json` --- JSON encoder and decoder > Encoders and Decoders)

Verdict: SUPPORTED - cites the JSONDecoder conversion table (array -> list).

## q13 (table_lookup)
Question: What strings does the %p directive produce in the en_US locale?

Answer: The %p directive produces 'AM' or 'PM' in the en_US locale.

Citations:
- python-3.12.7/datetime#0201 — python-3.12.7/datetime.rst:2505-2651 (:mod:`!datetime` --- Basic date and time types > :meth:`~.datetime.strftime` and :meth:`~.datetime.strptime` Behavior > Technical Detail)

Verdict: PARTLY - the answer is right, but the cited note (datetime#0201) does not contain AM/PM; the retained table row datetime#0188 does.

## q14 (multi_document)
Question: Are the timeout limits in requests and urllib.request.urlopen both measured in seconds and both applied to the connection attempt?

Answer: The timeout limits in requests and `urllib.request.urlopen` are both measured in seconds and both are applied to the connection attempt, but the timeout feature in `urllib.request.urlopen` only works for HTTP, HTTPS and FTP connections.

Citations:
- python-3.12.7/urllib.request#0060 — python-3.12.7/urllib.request.rst:658-669 (:mod:`!urllib.request` --- Extensible library for opening URLs > OpenerDirector Objects)
- requests-2.32.3/advanced#0049 — requests-2.32.3/advanced.rst:1093-1141 (Advanced Usage > Timeouts)

Verdict: SUPPORTED, INCOMPLETE - both in seconds is supported; omits the separate read timeout in requests.

## q15 (multi_document)
Question: Which exception does json.loads raise for invalid JSON, and which one does requests raise for r.json()?

Answer: json.loads raises a JSONDecodeError for invalid JSON, and requests raises a JSONDecodeError for r.json()

Citations:
- python-3.12.7/json#0032 — python-3.12.7/json.rst:619-628 (:mod:`!json` --- JSON encoder and decoder > Standard Compliance and Interoperability > Repeated Names Within an Object)
- python-3.12.7/json#0009 — python-3.12.7/json.rst:225-282 (:mod:`!json` --- JSON encoder and decoder > Basic Usage)

Verdict: PARTLY UNSUPPORTED - the requests half is not supported by any cited passage (only json.rst is cited).

## q16 (version_dependent)
Question: Are cafile and capath still recommended for urllib.request.urlopen on Python 3.12?

Answer: The `cafile` and `capath` parameters are still recommended for `urllib.request.urlopen` on Python 3.12, as they specify a set of trusted CA certificates for HTTPS requests and more information can be found in the `ssl.SSLContext.load_verify_locations` method.

Citations:
- python-3.12.7/urllib.request#0002 — python-3.12.7/urllib.request.rst:37-131 (:mod:`!urllib.request` --- Extensible library for opening URLs)

Verdict: NOT SUPPORTED, WRONG - says cafile/capath are still recommended; the docs deprecate them since 3.6.
