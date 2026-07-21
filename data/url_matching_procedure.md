# Procedure: Filling URLs in a course JSONL file

Goal: for each line in a course's jsonl, assign the `url` field to the lesson
page on `academy.towardsai.net` whose content corresponds to that line's
`name`.

Inputs you provide:

- `COURSE_SLUG` — the course path segment, e.g. `ai-business-professionals`
  or `agent-engineering`.
- `SEED_URL` — any lesson URL in that course, e.g.
  `https://academy.towardsai.net/courses/take/<COURSE_SLUG>/multimedia/<id>-<slug>`.
  This is just an entry point so the sidebar loads.
- `JSONL_PATH` — the course's jsonl file with `name` keys and (mostly) empty
  `url` keys.

Derived:

- `BASE = https://academy.towardsai.net/courses/take/<COURSE_SLUG>/`

## Why it's not a straight name match

Sidebar anchor text does **not** always equal the jsonl `name`. The sidebar
labels were edited after the content was authored, so the true lesson title
lives inside an `<iframe>` embedded in each course page. The iframe is served
from a per-course subdomain on `super.site` (e.g.
`ai-for-business-professionals.super.site`,
`agent-engineering.super.site`) and its URL slug is the original title the
jsonl was written against.

So the pipeline is:

1. Scrape every `(sidebar_label, lesson_url)` pair from the course sidebar.
2. Fuzzy-match jsonl `name` → `sidebar_label` to fill the easy cases.
3. For anything still unmatched, or any fuzzy match that looks suspicious,
   open the lesson page and read `iframe.src` — its slug is the real title.
4. Write the jsonl back, only mutating `url`.
5. Remove jsonl rows that end up with no URL (they belong to other courses).

## Phase 1 — Scrape the sidebar

Navigate to `SEED_URL` in Chrome. Expand every collapsed module (Teachable's
player lazy-renders only the active chapter):

```js
document.querySelectorAll('.course-player__chapter-item__header')
  .forEach(h => { if (!h.classList.contains('ui-state-active')) h.click(); });
```

Collect every lesson anchor. Teachable uses several media-type routes —
`multimedia`, `lessons`, `texts`, `quizzes`, `presentations`, `downloads`,
`videos` — and appends a badge like "MULTIMEDIA" to the anchor's visible
text. Strip it:

```js
Array.from(document.querySelectorAll('a'))
  .filter(a => new RegExp(
    `/courses/take/${COURSE_SLUG}/(multimedia|lessons|texts|quizzes|presentations|downloads|videos)/\\d`
  ).test(a.href))
  .map(a => ({
    title: a.innerText.trim()
      .replace(/\s+/g, ' ')
      .replace(/ (MULTIMEDIA|VIDEO|TEXT|QUIZ|LESSON|PRESENTATION|DOWNLOAD)S?$/i, ''),
    url: a.href,
  }));
```

De-duplicate by URL and save to a TSV, one `title|url` pair per line.

Sanity check: read the module headers to get the expected lesson counts and
verify the total matches:

```js
Array.from(document.querySelectorAll('.course-player__chapter-item__header'))
  .map(h => h.innerText.match(/(\d+)\s*\/\s*(\d+)\s*Completed/)?.[2])
  .filter(Boolean)
  .reduce((a, b) => a + +b, 0);
```

Gotcha: the Chrome `javascript_tool` MCP tool truncates any string result at
about 1024 characters. Stash the full list on `window.__pairs` and read it
back in small slices (`window.__pairs.slice(0, 8).join('\n')`).

## Phase 2 — Fuzzy match in Python

Walk the jsonl. For each row with an empty `url`, normalize both the row
`name` and the candidate sidebar titles (lowercase, straighten curly quotes,
collapse non-alphanumerics to spaces), then use `difflib`:

```python
m = difflib.get_close_matches(candidate, sidebar_keys, n=1, cutoff=0.6)
score = difflib.SequenceMatcher(None, candidate, m[0]).ratio() if m else 0
if score >= 0.72:
    assign(row, sidebar_url_by_key[m[0]])
```

Also try a "stripped" variant of each name that drops leading `(video)` /
`(video N)` markers and trailing `(new)` / ordinal suffixes like `1 / 2 / 3`
— course authors often annotate the jsonl name with these and the sidebar
doesn't.

Only mutate the `url` field on the json object. Re-dump with
`ensure_ascii=False` so non-ASCII round-trips cleanly.

At the end, print three lists — you need all of them for Phase 3:

- **Unmatched**: jsonl rows still empty after the pass.
- **Unused**: sidebar lessons no row claimed.
- **Low-confidence**: matches that landed just above the 0.72 cutoff, or
  where the token overlap between `name` and URL slug is thin.

## Phase 3 — Reconcile by hand (with help from the iframe)

### 3a — Obvious rephrasings

Walk the unmatched rows against the unused sidebar titles and add manual
pairings for anything that's clearly the same lesson worded differently.
Representative rewrites: "HR" ↔ "Human Resources and Recruitment",
"Quiz Questions" ↔ "Quiz Case Studies", smart quotes ↔ straight quotes,
"Why Providing Context Matters" ↔ "Providing Context to LLMs".

### 3b — Read the iframe for the rest

Any sidebar title still unused after 3a is probably a renamed lesson. Open
its URL in Chrome, wait a few seconds for the player to mount, and read:

```js
document.querySelector('iframe')?.src
```

The iframe src looks like `https://<course-subdomain>.super.site/<slug>`.
The `<slug>` is the true title in kebab-case — match it back against the
unmatched jsonl rows.

Do this sequentially: `navigate → wait ~3s → read iframe.src`. Fetching the
pages concurrently from the DevTools context tends to time out.

Some iframes point at YouTube rather than `super.site` — those lessons are
video intros; match them to whatever jsonl row represents the corresponding
script or video.

### 3c — Audit low-confidence fuzzy matches

For every already-matched row, compute token overlap between `name` and the
URL slug:

```python
overlap = len(name_tokens & slug_tokens) / len(name_tokens)
```

Anything under ~0.4 deserves a second look, **and** watch out for the
specific failure mode where a "Quiz" row gets mapped to the non-quiz sibling
lesson (or vice versa) because the tokens overlap heavily. Re-read the
iframe for any suspect. In past runs this pattern caught rows where
"Quizzes: AI in X" had been mapped to "Using LLMs in X" and
"Accounting & Finance Quiz" to "Sales Workflow Quiz".

Also audit rows whose `url` was already populated in the source file —
those values are not automatically correct.

## Phase 4 — Write back and verify

Rewrite the jsonl line-by-line, only ever mutating the `url` field, so
surrounding keys and formatting are preserved. Print:

- filled vs empty counts,
- how many sidebar lessons are still unused,
- **URL collisions** — group rows by their assigned `url` and report every
  URL claimed by more than one row, along with each row's `name`. Expected
  causes: "(Old)" / "(Tutorial version)" / "(new)" variants of the same
  lesson, renamed lessons whose previous name still lives in the jsonl, or
  a lesson whose content got chunked into multiple jsonl rows (identical
  `name`). Surface these so the human can decide whether to keep, merge, or
  drop each duplicate — don't silently deduplicate.

```python
from collections import Counter
counts = Counter(r['url'] for r in rows if r.get('url'))
dups = {u: n for u, n in counts.items() if n > 1}
for url, n in dups.items():
    print(f'{n}x {url}')
    for r in rows:
        if r.get('url') == url:
            print(f'  - {r["name"]}')
```

For a well-authored pair of sources you'd expect every sidebar lesson to be
claimed by exactly one jsonl row. If a sidebar lesson has no jsonl match,
it's almost certainly embedded from a different course (common on Teachable)
and can stay unused.

## Phase 5 — Clean up leftover rows

Any jsonl row still empty after Phase 4 belongs to another course or to
content that isn't shipped in this course anymore. If the file is meant to
be course-scoped, drop those rows:

```python
out = [l for l in open(JSONL_PATH) if json.loads(l).get('url')]
open(JSONL_PATH, 'w').writelines(out)
```

## Checklist

- [ ] Sidebar scraped, all modules expanded, `<title>|<url>` TSV saved.
- [ ] Sidebar lesson count matches sum of "X / Y Completed" across modules.
- [ ] Fuzzy pass run, unmatched / unused / low-conf lists printed.
- [ ] Manual rephrasings added.
- [ ] Iframe `src` read for every still-unused sidebar lesson.
- [ ] Quiz / non-quiz siblings audited.
- [ ] Pre-existing URLs in the source jsonl audited.
- [ ] Final counts: filled + empty == total; unused sidebar lessons == 0
  (or documented exceptions).
- [ ] URL collisions printed (rows sharing a `url`) and human-reviewed:
  "(Old)" / "(Tutorial version)" variants dropped or kept per request;
  identical-name chunk duplicates resolved.
- [ ] Empty rows dropped (if appropriate for the file's scope).

## Operational notes that tend to bite

- Chrome `javascript_tool` output is truncated at ~1024 chars.
- `super.site` iframes are cross-origin — you can only read `iframe.src`,
  not their DOM.
- Teachable sidebars lazy-render; always expand chapters before scraping.
- Curly quotes (`"..."`, `'...'`) and en-dashes in jsonl names break naive
  string compare; normalize aggressively.
- `difflib` loves "Quiz" vs "Quiz Scenarios" pairs — cutoff 0.72 is a
  starting point, not a guarantee.
