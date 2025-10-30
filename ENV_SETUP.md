# Environment Configuration

This project uses environment variables for configuration, with support for both local development and production deployment.

## Quick Start

1. **Copy the example file:**

   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your actual API keys:**

   ```bash
   # Open in your favorite editor
   nano .env
   # or
   code .env
   ```

3. **Your `.env` file is already gitignored** - it won't be committed to version control.

## Configuration Hierarchy

The application loads configuration in this order (highest priority first):

1. **Environment variables** (e.g., `export OPENAI_API_KEY=...`)
2. **`.env` file** (loaded via python-dotenv)
3. **Config files** (`config2.json`, `config.yaml`) - fallback for local development

## Required Variables

### For Streamlit Frontend:

- `OPENAI_API_KEY` - Your OpenAI API key for GPT-4o

### For Pipeline (if running data collection):

- `SCOPUS_API_KEY` - Your Scopus API key

## Optional Variables

- `OPENAI_MODEL` - Model to use (default: `gpt-4o`)
- `SCOPUS_QUERY` - Override the Scopus search query
- `OPENAIRE_QUERY` - Override the OpenAIRE search query
- `SCOPUS_COUNT` - Max results from Scopus (default: 25)
- `OPENAIRE_MAX_RESULTS` - Max results from OpenAIRE (default: 10)
- `DISABLE_SSL_VERIFICATION` - Set to `true` if behind corporate proxy (development only)

## AWS App Runner Deployment

When deploying to AWS App Runner, set these environment variables in the App Runner console:

```
OPENAI_API_KEY=your-key-here
SCOPUS_API_KEY=your-key-here
```

The application automatically detects when running in AWS and adjusts behavior (e.g., enables SSL verification).

## Security Notes

⚠️ **Never commit `.env` or `config*.json` files to version control**

- `.env` is already in `.gitignore`
- Keep your API keys secure
- Rotate keys if they're accidentally exposed
