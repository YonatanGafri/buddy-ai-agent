# Buddy - an AI study buddy

Buddy watches what a student opens in the browser and decides whether to leave
them alone, say something, or close the tab. It is a single LLM agent running a
hand-written ReAct loop: read the tab, look things up if looking helps, decide.

The student types in their own words - `opened youtube.com - 'lo-fi beats'`, a
pasted URL, a Hebrew sentence, half a thought. Nothing parses that before the
agent sees it. Working out which site it is, and whether it matters right now,
is the agent's job.

## What it decides

| Action | Effect |
| --- | --- |
| `allow` | Silent. No message reaches the student. |
| `nudge` | A short message. The tab stays open. |
| `lock` | Blocks this one navigation, with a reason. |

Any decision may set a `callback` - seconds until the agent wakes itself and
looks again. That is its only way to follow up, since it is otherwise called
only when the student writes.

## How it works

![Architecture](data/architecture.png)

The loop runs turns until the model returns a decision. There is no step limit;
the ceiling is 240s of wall clock, leaving headroom under Vercel's 300s cap.
Four exits: a decision, an agent error (no browsing event in the prompt), a
deadline forcing one summarizing turn, and a stuck-agent fallback that allows
rather than hanging the browser.

Five tools. Four read the same Supabase the GUI's context panel shows; the
fifth reads the open web:

- `read_calendar()` - what is due, and when
- `read_todo_list()` - what is pending and what is done
- `read_long_memory()` - what survives past today
- `rewrite_memory(scope, text)` - overwrites `short` or `long`
- `read_website(url)` - the page's `<title>` and meta description, nothing more

`read_website` exists because the domain is not the page. A broadcaster hosts a
physics podcast and a university hosts a sports section, so "it is an
educational site" is a guess about the domain rather than a fact about the tab.
The whole HTML stays inside the tool and two short strings come out, so a page
costs the model a sentence rather than a document.

The agent may ask for several tools in one turn:

```json
{"type":"tool_call","tools":[{"tool":"read_calendar","args":{}},
                             {"tool":"read_todo_list","args":{}}]}
```

The system prompt and the examples are resent on every reasoning turn, so
two reads in one turn cost one resend instead of two. The single-call shape is
unchanged and still accepted. In the trace each tool remains its own step - a
batch changes how many turns are paid for, not what the steps look like.

Short memory is inlined in the system prompt every turn rather than fetched, so
there is deliberately no tool that reads it. Given the choice, the model spent a
turn asking for what it could already see in 12 of 20 test runs.

Nothing in the prompt scripts which action to take, which tool to call, or when
to escalate. Those are judgment, and the agent makes them.

## Layout

```
api/index.py        the four endpoints (Vercel loads one top-level app)
api/agent/loop.py   the ReAct loop
api/agent/prompts.py system prompt + few-shot examples
api/agent/tools.py  the five tools
api/agent/memory.py Supabase reads and writes
api/agent/llm.py    OpenAI-compatible chat completions
public/index.html   the GUI, a single self-contained bundle
data/               seed data, the architecture diagram, recorded examples
supabase/schema.sql the three tables
```

## Endpoints

| Method | Path | Returns |
| --- | --- | --- |
| `POST` | `/api/execute` | `{status, error, response, steps}` |
| `GET` | `/api/team_info` | team name, students, group number |
| `GET` | `/api/agent_info` | purpose, description, prompt template, examples |
| `GET` | `/api/model_architecture` | the architecture diagram, `image/png` |
| `GET` | `/` | the GUI |

`POST /api/execute` takes `{"prompt": "..."}`. Every response carries the same
four top-level fields, and `steps` logs each turn as
`{module, prompt, response}` - the module names match the architecture diagram.

A prompt with no browsing event in it (`what should I work on?`) comes back as
`status: "error"` with the steps intact. That is the agent's judgment, not a
validation layer in front of it - the reasoning that reached "no tab here"
belongs in the trace like any other.

## Running it

```bash
pip install -r requirements.txt

export OPENAI_BASE_URL=https://api.llmod.ai/v1
export OPENAI_API_KEY=...
export BUDDY_MODEL=...
export SUPABASE_URL=https://<project>.supabase.co
export SUPABASE_SERVICE_KEY=...      # service_role, server side only

uvicorn api.index:app --reload
```

Then open http://127.0.0.1:8000.

Without Supabase configured the agent falls back to an in-process store, so it
runs but forgets everything between processes. Without an LLM endpoint it
returns `status: "error"` - there are no mocked decisions anywhere, because a
fabricated one is worse than a visible failure.

`SUPABASE_SERVICE_KEY` is read server side only and never reaches the GUI.
Every decision goes through `/api/execute`; the browser never writes.

The context panel is the one exception, and it reads directly. It fetches
`memory`, `calendar_events` and `todo_tasks` over PostgREST with a publishable
key, so the panel shows the same rows the agent's tools read rather than a copy
the server passes along. RLS allows `select` and nothing else - an insert comes
back `42501 row-level security violation`, and there is no update or delete
policy at all. See `supabase/schema.sql`.

## Deploying

Vercel, from `vercel.json`. Set the five environment variables above in the
project settings - not in the repo.
