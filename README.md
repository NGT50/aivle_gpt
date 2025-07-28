# aivle_gpt

This repository provides a Python script that uses the OpenAI API to generate detailed Korean patent application documents. The script `generate_patent.py` reads a patent summary and creates a specification following the format used in Korea (technical field, background art, description of the invention, and claims).

## Usage

Set the environment variable `OPENAI_API_KEY` with your API key and run:

```bash
python generate_patent.py --summary "요약 내용" --claims 3
```

Use the optional `--claims` argument to specify how many claims should be generated.

The resulting specification is printed to standard output.
