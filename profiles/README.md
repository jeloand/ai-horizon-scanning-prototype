# Profiles folder

A **profile** is a YAML file that tells the scraper **where** to pull content from and **what** policy keywords to look for.

# minimal example
feeds:
  "OECD AI Policy": "https://oecd.ai/rss"
  "ILO Research":   "https://www.ilo.org/rss"
keywords:
  - artificial intelligence
  - upskilling
  - labour market

| Field      | Description                                                                    |
| ---------- | ------------------------------------------------------------------------------ |
| `feeds`    | Key = human-readable label, value = RSS/Atom URL.                              |
| `keywords` | Case-insensitive list; a space means an exact phrase (e.g. `"platform work"`). |

**Run a specific profile**
python scripts/policy_signal_scanner_v3.py --profile my_profile

Place each new profile as `profiles/<name>.yml`.
The file name (without `.yml`) becomes the `--profile` argument.

Save, commit, and you’re good to go!
