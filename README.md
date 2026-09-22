# AI Document Assistant — beginner guide

Upload PDF, DOCX, TXT or Markdown files, ask a question, and inspect cited passages.
This is a learning project, not a guarantee of accurate answers or a production multi-user document platform.

## 1. What you downloaded

- `app.py`: complete Streamlit application, with separate functions for each stage.
- `requirements.txt`: Python libraries to install.
- `README.md`: this guide.

No trained model or API key is bundled. You do not need to train a model or buy a GPU.
Use a computer with internet access. The first installation and embedding-model download can take several minutes and significant disk space. A hosted deployment also downloads the embedding model.

## 2. Understand the workflow

1. **Extraction:** read text from the file. PDF page numbers start at 1.
2. **Chunking:** divide text into 1,000-character passages with 200-character overlap. These are characters, not tokens. PDF passages stay within their original page.
3. **Embeddings:** Sentence Transformers turns passages into numerical representations of meaning.
4. **FAISS:** stores those vectors in memory and searches by similarity. Normalized vectors and inner-product search implement cosine similarity.
5. **Keyword search:** matches meaningful query words, ignoring case and common English filler words. Repeated matches are capped to reduce repetition bias.
6. **Hybrid search:** combines the top 12 semantic and positive keyword results using Reciprocal Rank Fusion: each result contributes `1 / (60 + rank)`. The four highest combined results are used, without duplicates. These scores are not confidence probabilities.
7. **Generation:** Groq receives the question and selected excerpts, and writes an answer with source labels.

Only the embedding model is globally cached. Document text, metadata and the FAISS index stay in the current Streamlit session. Repeated questions reuse the index. A restart or lost session requires processing again. This is session reuse, not permanent storage or model training.

## 3. Easiest online route: GitHub + Streamlit

This route does not require installing Python on your laptop.

1. Extract `AI_Document_Assistant.zip` using Windows **Extract All**.
2. Open https://github.com and sign in or create an account.
3. Create a repository named `ai-document-assistant`.
4. Choose **Add file → Upload files** and upload `app.py`, `requirements.txt`, and `README.md` directly into the repository root. Commit the upload.
5. Open https://console.groq.com/keys and create an API key. Keep it private. Do not paste it into GitHub or app.py.
6. Open https://share.streamlit.io and connect your GitHub account.
7. Choose **Create app**, then select your repository, its actual branch (usually `main`), and `app.py` as the main file.
8. In **Advanced settings**, select Python 3.11 if available and paste the following into **Secrets**, replacing only the placeholder:

```toml
GROQ_API_KEY = "paste_your_real_groq_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
```

9. Deploy. Allow dependencies to install. Model loading happens when documents are first processed.
10. If already deployed, open the app's settings and update **Secrets** there. Save and restart if required.
11. Open the app URL, upload a small text document, click **Process Documents**, then ask a question.

Groq lists `openai/gpt-oss-120b` in its official documentation at the time this project was prepared. Availability, account access and rate limits can change. If necessary, replace `GROQ_MODEL` with a currently supported text chat model from https://console.groq.com/docs/models.

## 4. Local Windows setup, step by step

Use this route if you want the app running on your own computer.

### A. Install Python and open the folder

1. Install 64-bit Python 3.11 from https://www.python.org/downloads/ and select **Add Python to PATH** during installation.
2. Extract the ZIP. Open the folder containing `app.py`.
3. Click the File Explorer address bar, type `cmd`, and press Enter. This opens Command Prompt in the correct folder.
4. Check Python:

```bat
py -3.11 --version
```

If `py` is unavailable but `python --version` shows Python 3.11, replace `py -3.11` with `python` in the next command.

### B. Create a virtual environment and install packages

Run each line separately:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A virtual environment keeps this project's packages separate from other projects. You should see `(.venv)` in the terminal. The install may download PyTorch and other large packages; do not close the terminal halfway through.

### C. Add your Groq key

In the same Command Prompt:

```bat
mkdir .streamlit
notepad .streamlit\secrets.toml
```

Accept creating the file, paste the following, replace the placeholder and save:

```toml
GROQ_API_KEY = "paste_your_real_groq_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
```

Make sure the filename ends in `.toml`, not `.toml.txt`. Do not upload this file to GitHub.
An environment variable named `GROQ_API_KEY` is also supported and takes priority over secrets.

### D. Start the app

```bat
python -m streamlit run app.py
```

Your browser should open automatically. Otherwise use the local URL printed in the terminal, usually http://localhost:8501.

To stop, press **Ctrl+C** in the terminal. Next time, open Command Prompt in this folder, activate `.venv` and run the same Streamlit command.

### E. Before using Git to upload the whole folder

Create a file named `.gitignore` and add:

```gitignore
.venv/
.streamlit/secrets.toml
.env
__pycache__/
*.pyc
```

Also keep personal documents out of the repository. Ignoring a key does not remove an already committed key: revoke and replace exposed keys through Groq.

### macOS/Linux alternative

Use a Python 3.11 installation and run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p .streamlit
```

Create `.streamlit/secrets.toml` in a text editor with the same contents as above, then run `python -m streamlit run app.py`.

## 5. Your first test: use fictional data

Create a UTF-8 text file named `demo_policy.txt` in Notepad with:

```text
Demo Company Policy (fictional)
Employees receive 20 days of annual leave each calendar year.
Leave requests must be submitted to the manager at least 7 days in advance.
Policy code HR-LEAVE-07 covers annual leave.
Remote work is permitted on Fridays with manager approval.
```

Upload it and click **Process Documents**. After processing:

| Question | What to check |
|---|---|
| How many days of annual leave are allowed? | 20 days, supported by a source label |
| How much time off can staff take each year? | Paraphrased wording still finds annual leave |
| What does HR-LEAVE-07 cover? | Exact identifier helps keyword search |
| What is the CEO's salary? | Says the information could not be found |
| Can I work remotely every day? | Does not expand the Friday permission to all days |

Then try these checks:

- Ask a second question: it should not rebuild document embeddings.
- Click Process Documents again with the same files: it should reuse the index.
- Change a file or chunk setting: questions should be disabled until reprocessing.
- Change the question: the old answer should disappear rather than appear to answer the new question.
- Upload two files: check sources refer to the correct filename.
- Upload a readable PDF with a fact on page 2: check its citation says page 2.
- Upload a blank file alongside a valid file: the valid file should still process.
- Clear documents: uploaded files, index and answers should reset.

## 6. Where to learn from the code

Read the functions in this order:

1. `extract_pdf`, `extract_docx`, `extract_txt`, `extract_md`
2. `make_chunks`
3. `load_embedder` and `build_index`
4. `keyword_search`, `fuse_rankings`, `hybrid_search`
5. `answer_question`
6. `main` for the interface and session-state behavior

The default chunk settings are defined in the sidebar sliders. The embedding model is `EMBED_MODEL`; the default answer model is `DEFAULT_GROQ_MODEL`. The secrets setting overrides the default answer model.

## 7. Limits and privacy

- Scanned PDFs need OCR; OCR is not included. Image-only pages in mixed PDFs are skipped with a notice.
- DOCX extracts body paragraphs and tables in order. Headers, footers, text boxes and embedded images are not included.
- DOCX, TXT and Markdown do not have reliable page numbers here.
- PDF columns and complex layouts may extract in imperfect reading order. Inspect the preview.
- The MiniLM embedding model and English stop-word list are best suited to English. Urdu questions may be answered, but multilingual retrieval is not guaranteed.
- Chunking uses character windows. The embedding model can truncate long token sequences, particularly for dense or non-English text; smaller chunks may help.
- Limit: 25 MB combined upload and 2,500 chunks. These are learning-app guardrails, not guarantees of capacity on every host.
- Documents are processed on the machine hosting Streamlit. The question and four selected excerpts are sent to Groq. Do not upload confidential content without appropriate permission.
- No document database, login or permanent history is included. Protect access and API spending before sharing widely.
- The model is instructed to ignore commands inside documents, but this does not make it immune to prompt injection.
- Retrieval always returns its best candidates, including for unrelated questions. The language model must judge sufficiency; unsupported-answer rejection is not guaranteed.
- Source labels are checked for valid numbers, not automatically verified for factual entailment. Always read the actual passages.
- Dependency ranges are bounded rather than an exact lockfile. Once installed successfully, you can record your environment with `python -m pip freeze > requirements-lock.txt`.

## 8. Troubleshooting

| Problem | Action |
|---|---|
| Python is not recognized | Install Python with PATH enabled, reopen Command Prompt, or use the `py` launcher |
| Missing package / ModuleNotFoundError | Activate `.venv`, then rerun `python -m pip install -r requirements.txt` |
| PowerShell blocks activation | Use Command Prompt and `.venv\Scripts\activate` as shown above |
| pip cannot find a compatible FAISS wheel | Use 64-bit Python 3.11 and an updated pip |
| Embedding model cannot load | Check network access to Hugging Face and available RAM/disk; retry processing after fixing access |
| Missing Groq key | Confirm `.streamlit/secrets.toml` locally or app Secrets in Cloud; check exact name `GROQ_API_KEY` |
| Authentication error | Verify the key in Groq; replace revoked or incorrect keys |
| Model request fails | Check official supported models and your account's access, then update `GROQ_MODEL` |
| Rate limit | Wait, ask shorter questions, or check account limits; the app does not retry indefinitely |
| No text extracted | Try a selectable-text PDF or UTF-8 TXT; unlock password-protected PDFs; OCR scanned pages elsewhere |
| Memory limit on Cloud | Upload fewer documents or smaller files; restart and reprocess |
| Wrong answer | Inspect retrieved sources, rephrase the question, or reduce chunk size; do not assume citations prove correctness |

## 9. Validation of this delivered version

Completed checks: Python compilation; real PDF extraction and page metadata; DOCX paragraph/table order; UTF-8 TXT and Markdown extraction; chunk overlap and invalid settings; keyword edge cases; rank fusion; actual FAISS indexing/search with deterministic test embeddings; and Streamlit initial render with pre-index question controls disabled.

Not completed: downloading/running the actual Sentence Transformers model, live Groq calls, end-to-end answer accuracy, or a Cloud deployment. Those require validation in your environment. No API key is included.

## Official documentation

- Streamlit installation: https://docs.streamlit.io/get-started/installation
- Streamlit cloud deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Streamlit secrets: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- Groq models: https://console.groq.com/docs/models
- Selected model: https://console.groq.com/docs/model/openai/gpt-oss-120b
- Sentence Transformers model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
