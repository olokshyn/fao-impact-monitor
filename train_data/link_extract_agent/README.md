# LinkExtractAgent train data

HTML pages and gold URL lists used to train and evaluate `LinkExtractAgent`.

- `examples.json` — topic, data_filter, page URLs, gold link lists
- `*.html` — downloaded page bodies (do not re-fetch in tests)

Train:

```bash
uv run train-link-extract-agent
```

Note: the emergencies page pagination gold URL is the relative `href` as it
appears in the HTML (after entity unescaping). The absolute form with a full
query string is not present as a literal substring in the page source.
