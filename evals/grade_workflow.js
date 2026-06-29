export const meta = {
  name: 'subagent-judge-grading',
  description: 'Blinded subagent judge grades holistic/faithfulness/key-point/behavior chunks',
  phases: [{ title: 'Grade', detail: 'one blinded grader agent per chunk' }],
}

// args: { battery, chunks: <int> }  OR  { battery, indices: [<int>...] }
// indices lets us re-run only the missing chunks after a partial run.
let A = args
if (typeof A === 'string') {
  try { A = JSON.parse(A) } catch { A = {} }
}
const battery = (A && A.battery) || 'singleturn'
const explicit = A && Array.isArray(A.indices) ? A.indices : null
const nChunks = (A && A.chunks) || 0
if (!explicit && !nChunks) throw new Error('pass args.chunks (count) or args.indices (array)')

const dir = `runs/_grading/${battery}`

const SUMMARY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    chunk: { type: 'integer' },
    graded: { type: 'integer', description: 'rows graded and written' },
    low_confidence: { type: 'integer', description: 'count of confidence=low verdicts' },
    wrote: { type: 'string', description: 'path of the verdicts json written' },
  },
  required: ['chunk', 'graded', 'low_confidence', 'wrote'],
}

function prompt(i) {
  const pad = String(i).padStart(3, '0')
  const chunkPath = `${dir}/chunk_${pad}.csv`
  const outPath = `${dir}/verdicts_${pad}.json`
  return `You are a BLINDED grader for an AI tutor on a course platform (applied AI, LLMs, RAG, Python).

INPUT: parse the CSV at ${chunkPath} with Python (use csv with field_size_limit(10_000_000); fields contain newlines). Columns: sheet_row_id, item_type, question, criterion, reference, answer.

RUBRICS: read the dict RUBRICS and the string PREAMBLE in evals/judge.py and follow them EXACTLY, selecting the rubric by each row's item_type (for "probe:*" use the "probe" rubric; for "faithfulness" the 'reference' field IS the retrieved evidence to grade grounding against; for unknown types use "key_point"). These are the same rubrics the project's validated judge uses.

RULES:
- Grade EVERY row in the chunk. You see only content; you do NOT know which system/preset produced each answer and must not infer or use it.
- An EMPTY answer = fail (confidence high).
- grade is "pass" or "fail"; always decide. confidence is "high" or "low" — use "low" ONLY for genuinely borderline calls a reasonable grader could see either way. reason = one short sentence.

OUTPUT: write a JSON file to ${outPath} that is a list of objects, one per row, each: {"sheet_row_id": <str>, "item_type": <str>, "grade": "pass"|"fail", "confidence": "high"|"low", "reason": <str>}. Write it with Python json.dump. Every input sheet_row_id must appear exactly once.

Then return the summary (chunk index ${i}, how many you graded, how many were low confidence, and the output path). Your returned text is parsed as data, not shown to a human.`
}

phase('Grade')
const indices = explicit || Array.from({ length: nChunks }, (_, i) => i)
const results = await parallel(
  indices.map((i) => () =>
    agent(prompt(i), { schema: SUMMARY_SCHEMA, label: `${battery}:chunk_${i}`, phase: 'Grade' })
  )
)

const ok = results.filter(Boolean)
const graded = ok.reduce((a, r) => a + (r.graded || 0), 0)
const low = ok.reduce((a, r) => a + (r.low_confidence || 0), 0)
const failedChunks = indices.filter((_, k) => !results[k])
log(`${battery}: ${ok.length}/${indices.length} chunks done, ${graded} rows graded, ${low} low-confidence`)
if (failedChunks.length) log(`FAILED chunks (rerun): ${failedChunks.join(', ')}`)
return { battery, chunks_done: ok.length, chunks_total: indices.length, graded, low_confidence: low, failed_chunks: failedChunks }
