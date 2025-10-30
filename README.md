## Clinical Note Evaluation Suite

**Nathan Lim**

---

An evidence-grounded, accuracy-focused evaluation suite for comparing LLM-generated SOAP notes against clinical transcripts. It performs claim-level fact checking via RAG + LLM-as-judge, flags hallucinations (unsupported/contradicted claims), and identifies clinically critical information that is missing from the note.

### Main features

- Claim-by-claim verification with explicit citations and rationale to the transcript
- Hallucination breakdown (Unsupported vs. Contradicted) with severity
- Detection of clinically critical content missing from the note
- Parallelized calls for speed, careful rate limiting, and token accounting
- A simple Streamlit dashboard for aggregate visualization

---

### Installation and setup

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Provide your OpenAI API key (note: you do not need to provide an API key to run the dashboard)

```bash
export OPENAI_API_KEY=your-api-key
# or create a .env file in the root directory with:
# OPENAI_API_KEY=your-api-key
```

Optionally configure rate limits in your .env (defaults are set to only use up to tier 1 RPM limits, but system can benefit from improved latency with a higher tier):

- `OPENAI_RPM_LIMIT` - general rpm limit
- `OPENAI_RPM_LIMIT_RESPONSES` - limit for responses; overrules general limit,
- `OPENAI_RPM_LIMIT_EMBEDDINGS` - limit for embeddings; overrules general limit
- `OPENAI_TPM_LIMIT` - general tpm (tokens per minute) limit
- `OPENAI_TPM_LIMIT_RESPONSES` - limit for responses; overrules general limit
- `OPENAI_TPM_LIMIT_EMBEDDINGS` - limit for embeddings; overrules general limit

---

### Quickstart

- Models: defaults to `gpt-4.1-mini` for judging and `text-embedding-3-small` for retrieval
- Paths: relative or absolute paths are both supported

#### To run for a single transcript/note pair

```bash
python evaluate_note.py \
  --transcript test-data/example-transcript.txt \
  --note test-data/example-note.txt \
  --out results/individual_result.json
```

Key options (optional to modify; I suggest leaving as defaults):

- `--k`: top-k transcript windows per claim (default: 5)
- Retrieval chunking controls: `--retrieval-chunk-min-chars`, `--retrieval-chunk-max-chars`, `--retrieval-chunk-min-sents`, `--retrieval-chunk-max-sents`, `--retrieval-response-min-sents`
- Concurrency: `--judge-concurrency`, `--criticality-concurrency`, `--verification-concurrency`
- Verification: `--verification-k` top-k note claims for missing-critical verification

#### To run for a batch of transcript/note pairs

Prepare a CSV with at least `transcript,note` columns (optional: `id,out`). Example (`test.csv`):

```csv
id,transcript,note,out
0,tests/transcript1.txt,tests/note1.txt,results/batch-results/0.json
1,tests/transcript2.txt,tests/note2.txt,results/batch-results/1.json
```

Run the batch evaluator (writes per-item JSON to `results/batch-results` and creates an aggregate JSON in `results/batch_results_aggregated.json`):

```bash
python evaluate_batch.py \
  --pairs-csv test-data/test.csv \
  --out-dir results/batch-results \
  --aggregate-json results/batch_results_aggregated.json
```

Other command line argument flags largely mirror `evaluate_note.py` and are passed through to the evaluator.

#### Run the web app (dashboard)

To view a dashboard summarizing data findings from a batch eval, first make sure you've run evaluate_batch.py and have identified the directory that the resulting JSON files are stored.

Run the command below, replacing `results/batch-results` with the directory that contains the stored JSON files.

```bash
streamlit run webapp.py -- \
  --evals-dir results/batch-results
```

The command automatically launches a Streamlit dashboard and will print a local URL to find the dashboard.

Navigate to the local URL to explore aggregate metrics and distributions.

---

### How the evaluation works

The pipeline in `note_eval/pipeline.py` performs the following steps:

1. Claim extraction from the note

   - Split the note into sentences (pysbd) and section-aware fragments (S/O/A/P hints)
   - Filter trivial/enumeration-only fragments and short text; what remains are "claims" from the note (in reality, these are just sentences from the note--but since they are sentences in a note, they are essentially guaranteed to have one or more true claims)

2. Transcript preparation and conversation-aware chunking

   - Sentence-level indexing with character spans
   - Chunk the transcript by pairing related questions and answers within chunks and segmenting boundaries at new questions to efficiently and effectively chunk by relevance

3. Embedding and retrieval (RAG)

   - Embed chunks with OpenAI `text-embedding-3-small`
   - For each claim, retrieve top-k most similar transcript windows

4. LLM-as-judge for each claim

   - Build a prompt with locally numbered evidence sentences (numbered so the LLM can use specific sentence numbers as evidence citations)
   - The judge returns one label: Supported, Contradicted, or Unsupported
   - Also returns citations and a severity level (high/medium/low; none for Supported)

5. Coverage mapping

   - All of the citations received from the judge are mapped back to global transcript sentence indices to compute what was “covered” by supported claims

6. Criticality for uncovered transcript sentences (missing-only)

   - After the mapping, uncovered transcript sentences are at risk of containing missing critical information! But much of it is just conversational pleasantry as well. To filter, a new LLM binary classifier marks whether each uncovered sentence is clinically critical for this encounter

7. Verification for critical-but-missing segments

   - It is possible that uncovered segments are so similar to other portions of the transcript (and the note) that the judge LLM may not have had a chance to cite it; even if it is clinically relevant, it would be inaccurate to state that the note is missing the information because there are many cases where the note already covers very similar information
   - To address this, we run another check for critical-but-missing segments by running a reverse retrieval: for every critical-but-missing candidate segment, we retrieve top-k most similar claims **from the note** and ask the LLM if all the information in the critical-but-missing segment is captured sufficiently in the note
   - If the LLM's answer is no, then we have confirmed that the segment is both clinically relevant AND missing from the final note--in this case, the LLM also assigns a severity to the missing critical segment

8. Aggregation and outputs

   - Per-claim arrays and a `summary` block with key counts and rates
   - Token usage aggregated via the shared `OpenAIClient`

9. Summarization of Findings (unrelated to the pipeline in `note_eval`)
   - After all evaluation data is retrieved, `webapp.py` can be run to generate a simple yet informative dashboard containing summaries of the results from the evals. This dashboard includes statistics like the total number of claims, the probability of a claim being hallucinated, how many missing critical facts there were, the severity levels of hallucinations and missing critical facts, and so on. It also shows the note with the highest rate of hallucinations, along with its transcript and LLM rationales for why each of the hallucinated claims are hallucinations. Lastly, it gives information about the run's usage of OpenAI's API, including the token counts, calls, and total cost generated by the entire evaluation batch run.
   - All of this information would help to very quickly understand production quality of newly generated SOAP notes. The system does not require any ground truth labels for the evaluation process itself, so it could be deployed in a sort of random monitoring sense to catch specific areas where notes may have lower quality that might not have been explicitly tested. Not requiring ground truth labels also helps greatly with the goal of moving fast and without delay (which would be the case if we were forced to wait for GT labeled data on every evaluation)

### File overview

- Top-level CLI and app

  - `evaluate_note.py`: Runs the evaluator for a single transcript/note pair and writes one JSON output.
  - `evaluate_batch.py`: Batch runner over a CSV of pairs; parallelizes evaluations, writes per-item JSONs, and produces an aggregate JSON.
  - `webapp.py`: Streamlit dashboard to explore batch results (counts, rates, distributions) and aggregated token usage.

- Core library (`note_eval/`)
  - `pipeline.py`: Orchestrates the full evaluation: claim extraction, retrieval, LLM judging, coverage mapping, criticality, and verification; returns summaries and detailed arrays.
  - `preprocess.py`: Text utilities: normalization, sentence segmentation (pysbd), SOAP section detection, `note_to_claims`, `transcript_sentences`, and `number_sentences_for_prompt`.
  - `retrieval.py`: Conversation-aware chunking, embedding index construction, and top‑k evidence window retrieval per claim.
  - `judger.py`: Builds judge prompts and executes the LLM-as-judge call; yields label, citations, rationale, and severity.
  - `coverage.py`: Maps judge citations back to global transcript sentence indices to compute covered sentences.
  - `criticality.py`: Binary classifier prompts/calls to mark uncovered transcript sentences as clinically critical for this encounter.
  - `verification.py`: Reverse-RAG check against the note: retrieves similar note claims for each critical segment and verifies presence; assigns severity if missing.
  - `openai_client.py`: OpenAI API wrapper for embeddings and responses with RPM/TPM throttling, retries, and token usage accounting.

### Design notes

- Near Deterministic behavior: all calls use temperature 0.0
- High parallelism: separate thread pools for judging, criticality, and verification--as well as separate threads for individual evals when running `evaluate_batch`. This massively improves latency but is limited primarily by the API usage tier from OpenAI for specific RPM limits that must be kept beneath, which is especially limiting given the system is designed around a very large number of very small calls. As a result, there is very careful RPM/TPM handling to stay within API limits. Constrained by default API limits, every 10 note/transcript pair evaluations equate to approximately 70 seconds additional latency in `evaluate_batch.py`.
- Cheap: despite being built around LLM-as-a-judge evaluation, use of GPT-4.1-mini and the deliberate design of prompts to add additional input tokens over additional output tokens (ex. using numeric sentence label citations instead of direct quotes) keeps API costs very low especially given that input tokens are ~5x cheaper than output tokens. A single note/transcript pair evaluation costs approximately $0.011 using this dataset.

---

### Outputs

Single run (`evaluate_note.py`) writes a JSON with:

- `summary`: counts by category and severity (claims, hallucinations, unsupported, contradicted, missing_critical, etc.)
- `token_usage`: approximate token accounting
- `claims`: all claims with retrieval windows and judge outputs
- `hallucinated`, `contradicted`, `missing_critical`: filtered subsets for convenience

Batch run (`evaluate_batch.py`) additionally writes:

- Per-item JSONs in `--out-dir`
- Aggregate JSON (via `--aggregate-json`, defaults to `results.json`)

---

### Configuration reference

- Core
  - `--model` (default: `gpt-4.1-mini`)
  - `--api-key` (fallback to `OPENAI_API_KEY`)
- Retrieval
  - `--k` (default: 5), `--embedding-model` (default: `text-embedding-3-small`)
  - `--retrieval-chunk-min-chars` (160), `--retrieval-chunk-max-chars` (1000)
  - `--retrieval-chunk-min-sents` (2), `--retrieval-chunk-max-sents` (8)
  - `--retrieval-response-min-sents` (2)
- Claim extraction
  - `--min-claim-chars` (12)
- Concurrency
  - `--judge-concurrency` (20), `--criticality-concurrency` (20), `--verification-concurrency` (20)
- Missing-critical verification
  - `--verification-k` (5)

---

### Existing Test Data and Sample Outputs

This repository includes small, ready-to-run inputs and example outputs so you can try the suite without preparing your own data.

#### Test data (`test-data/`)

- `test.csv`: test dataset via adesouza1/soap_notes dataset that I primarily used for testing and for all the examples and sample outputs. This was one of the datasets specified in the assessment description.
- `example-transcript.txt`: short, realistic clinical conversation transcript from the train split of the above dataset
- `example-note.txt`: example SOAP-style note corresponding to the above transcript

Reproduce on the provided examples:

```bash
python evaluate_note.py \
  --transcript test-data/example-transcript.txt \
  --note test-data/example-note.txt \
  --out results/individual_result.json

python evaluate_batch.py \
  --pairs-csv test-data/test.csv \
  --out-dir results/batch-results \
  --aggregate-json results/batch_results_aggregated.json
```

#### Sample outputs (`results/`)

- `individual_result.json`: single-run output for the example pair
- `batch-results/`: per-item outputs when running the batch example (e.g., `results/batch-results/0.json`, `1.json`, ...)
- `batch_results_aggregated.json`: aggregate metrics over the batch run (mirrors the summary fields and adds simple distribution info)

**Because there are sample outputs that correspond very closely to the data you would get if you ran the evaluation yourself, feel free to skip running the evaluation if you are only interested in seeing the dashboard. Simply run it with the code below:**

```bash
streamlit run webapp.py -- \
  --evals-dir results/batch-results
```

### Note about evaluation results

I believe that this dataset--if I understand the assessment instructions correctly--was originally intended to act as a source of ground truth clinician-edited SOAP notes; I could not find whether these SOAP notes were real or synthesized, however, I found that these SOAP notes introduce many facts that are _not_ present in the transcript--most of this comes down to the SOAP note referencing some sort of physical/visual exam or other such cues that a clinician would be able to take note of, but an LLM-generated SOAP note would never be able to write about (assuming the LLM also only has access to the conversation transcript). Therefore, I decided to treat this dataset as faulty-data--I treated anything that wasn't discernible from the transcript alone as a "hallucination". If these _were_ LLM-generated SOAP notes, that is exactly how I would approach it. There are also some flat out mistakes in this dataset, ex. noting that a patient has no swelling when the transcript clearly says they noticed puffiness. These were marked as hallucinations via contradicted claims (whereas the other hallucinations were via unsupported claims). To experiment with prompt engineering and various models, I manually went through a few examples to determine what should be considered a hallucination and compared my own analysis with the evaluation framework. **This is also, at the end of the day, how this evaluation framework would need to be validated.** I would find it hard to imagine a way to validate the evaluation framework without a small set of manually annotated ground truth examples--however, it would be rather easy to test on a more granular level because you could simply use the prompt-building functions to build similar prompts that the model would be receiving (with pre-planned retrieved transcript spans and claims), which almost comes close to "unit-testing" the system in a way. Given a set of specific spans of evidence and a specific claim, you would be validating whether the LLM judge comes to the same conclusion as the ground truth--which is either that the claim is supported, unsupported, or contradicted by the evidence given. Increased determinism is a huge part of what makes this work, and I will touch on that more below.

---

#### Tradeoffs

I considered a number of different approaches to the problem. One of the biggest decisions I made was to go with an approach entirely based around LLM-as-a-judge instead of more deterministic metrics. I decided to do this because I think that deterministic metrics cannot be relied upon for this type of evaluation because detecting errors from LLMs inherently requires human-like understanding of language. Deterministic metrics like comparing n-grams from a generated SOAP note to a ground truth clinician edited one are going to be very unreliable in how well they actually reflect the LLMs' performance (and errors).

Specifically, string and overlap metrics like BLEU and other types of n-gram/token overlaps wouldn't work well because they rely too much on exact wordings. In conversation/note-taking, especially in healthcare where there are many synonyms for different terms, relying on the presence of exact words alone won't be accurate enough for any helpful evaluation insights. Trying to use similarity-based metrics like cosine similarity on embeddings alone (without any LLMs, for example) is also very risky because semantic similarity does not equate to factual entailment, especially when it comes to hallucinated small details like dosages and durations. Rule-based metrics were another thing I looked into, but there are simply too many different phrases, terms, and possible ways to say the same thing that it would be nearly impossible to implement it quickly and effectively. There are some LLM-_like_ approaches, like NLI models for detecting entailment between claim and evidence, but they also do not have high enough accuracy for a task like this. Accuracy among all of these deterministic metrics is poor because there is simply too much ambiguity when it comes to what may be said in a clinician-patient dialogue and how it may be said.

After all of this consideration, I came to the final conclusion to approach the problem using a pure LLM-as-a-judge approach. I also immediately had two priorities in mind: token usage and latency. Minimizing token usage/overall cost of running the evaluation system is important so that we can understand production quality without worrying about running up the bill constantly on an evaluation system; minimizing latency is important so that we can continue to move fast and quickly iterate through various designs and new models.

At first glance, the easiest way to implement this is to take each transcript and generated SOAP note and throw it all into an LLM with a good prompt. There are a few problems with this approach. While it works well for very big errors between the note and the transcript, asking the LLM to find small discrepancies between the entire note and the entire transcript at once (ex. the note refers to 600mg of a medication and the transcript mentions the medication but not the amount) was very inconsistent in my testing. With top-of-the-line models like GPT-5 Pro, missing hallucinations and any excluded critical findings was far less common, but latency and token expense was a massive issue. Another problem with GPT-5 Pro, was also that it would tend to have many false positives trying to flag hallucinations; my hypothesis is that it would start to _overthink_ what should be considered sufficient or not because of its heavy emphasis on constant reasoning. Testing with base GPT-5 gave inconsistent results (sometimes would miss many hallucinations, sometimes would flag far too many false positives), which was something I was trying to avoid because my goal of using LLM-as-a-judge was to have both accuracy and consistency in the evaluations.

My solution uses RAG with GPT-4.1-mini to solve the latency, token expense, and consistency issue. By isolating the LLM calls to very small, specific tasks ("is everything in this claim backed up by evidence from these spans in the transcript?"), consistency and accuracy greatly improves. GPT-4.1 architecture specifically allows for even greater consistency because it is the latest model from OpenAI **that still supports the temperature parameter**. This allows me to explicitly set the model temperature to 0.0 and thus makes the LLM outputs almost completely deterministic. This helps to make the evaluations much more consistent (same input -> same results), which is a key part of creating an effective evaluation because if you re-run an evaluation, you want to make sure that the results improved because your input improved--not because of random chance in the evaluation itself. Having greatly increased determinism also helps to ensure that we are able to validate that the evaluation system is evaluating input as expected. As explained above in the evaluation results section, validation testing of an evaluation system, especially in a more "unit-test" like format, only works if output is primarily deterministic--which we achieve with GPT-4.1. Using GPT-4.1-mini also helps greatly to decrease latency and token expense. GPT-4.1's architecture is one of the fastest in OpenAI's model offerings and GPT-4.1-mini is one of the cheapest. At first glance, GPT-5 mini appears to be a better alternative because by pure token usage, it is cheaper, but it would not allow us to set the temperature parameter as mentioned previously and also uses reasoning. This means that, unlike GPT-4.1, GPT-5 will use up to double the amount of tokens that GPT-4.1 would use on the same prompt with the same length of output. This is true even with the reasoning parameter set to "minimal".

Within the adesouza1/soap_notes dataset which was linked in the assessment description, my solution has an average API cost of $0.011 per single note evaluation, which is far less than the costs I was seeing when I tested the system with GPT-5-mini or when I provided the transcript and SOAP note in a single prompt evaluation to regular GPT-5.

For RAG, there was also the question of which embeddings to use. I tested with an open source sentence transformers model first, namely MiniLM-L6-v2 from Hugging Face. This didn't give the best performance for retrieval within the system and was consistently ranking relevant pieces of information as dissimilar. Because of this, I decided to pivot to using OpenAI's Text Embedding 3 (small). Because of the relatively tiny (in comparison to, say, textbooks) size of these conversation transcripts, the API cost for using the embeddings is so minuscule it still appears as <$0.01 in my billing section for my OpenAI account as I write this (_after_ having tested the system with hundreds of notes). The gain in performance was at nearly zero cost, so I decided to stick with OpenAI's embeddings. For chunking size, I decided to go with algorithmic chunking based off of an analysis of questions and answers within the transcript instead of a set size. I did this because after analyzing many of the transcripts in the available data, I came to the realization that most of these conversations are in a pretty standard format--either the clinician asks a question, and the patient gives an answer, or the patient asks a question, and the clinician gives an answer. In both cases, the answer is linked to the question and both are essentially linked pieces of information. I chunk the transcript based off of this format so that each chunk maintains question-answer pairs for relevancy in retrieval.

My final implementation, after having weighed and evaluated all of these different approaches, is a solution that focuses on transparency, reproducibility, and scalability; each decision is traceable via explicit evidence/rationales and evaluation can be parallelized efficiently across many transcript-note pairs in a token-efficient manner.
