# Config folder

Only **sample** files live here – they show the expected keys & shapes.
Copy one, rename, and add your real secrets *locally*.

cp config.sample.yaml config.yaml # SCOPUS key, etc.
cp config2.sample.json config2.json # OpenAI key


| File                   | Purpose                        |
|------------------------|--------------------------------|
| `config.sample.yaml`   | YAML template for SCOPUS / misc API keys. |
| `config2.sample.json`  | JSON template for OpenAI key.  |

> ⚠️  **Never commit real credentials.**  
> The root `.gitignore` already excludes `config.yaml`, `config.json`, and any non-sample files in this folder.
