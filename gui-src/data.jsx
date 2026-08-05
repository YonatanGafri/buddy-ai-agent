/* Buddy - sim clock + backend client.

   The GUI holds no browsing model of its own. There are no seed tabs, no
   domain list and no classification here - the one tab on screen comes out of
   the agent's decision. What is left is:
     - BUDDY_CONFIG: Supabase + execute endpoint, overridable via window.BUDDY_CONFIG
     - fetchContext(): pull memory/calendar/todo from Supabase REST (anon read)
     - runExecute(): POST the student's prompt to /api/execute
   fetchContext degrades to local seeds so the panel still renders with no
   Supabase; runExecute has no fallback on purpose - see below. */

/* ---- backend config ----
   Set window.BUDDY_CONFIG before this script to point the GUI at a different
   Supabase project or backend. */
const BUDDY_CONFIG = Object.assign({
  supabaseUrl: 'https://jneldbbbiaamhtsyflso.supabase.co',
  supabaseAnonKey: 'sb_publishable_1WfVJwB_Tbh1sePyw74cgQ_aVFu3Sve', // public; RLS is read-only
  // relative by default: the GUI talks to whatever host serves it, so the same
  // build works locally, on a preview deploy and in production.
  executeUrl: '/api/execute',
}, (typeof window !== 'undefined' && window.BUDDY_CONFIG) || {});

/* ---- what the panel shows when Supabase is unreachable ----
   Deliberately empty rather than a copy of data/calendar.json + data/todo-list.json.
   That copy existed, drifted from the real files, and made an outage look like a
   working panel showing stale coursework. An honest blank is better: this is the
   context the agent reads, and if we cannot read it, saying so is the true
   answer. Never reaches the agent either way - it reads the rows via its tools. */
const FALLBACK = {
  short: '', long: '', calendar: [], todo: { pending: [], completed: [] },
};

const cfgReady = () => !!(BUDDY_CONFIG.supabaseUrl && BUDDY_CONFIG.supabaseAnonKey);

async function sbGet(path) {
  const r = await fetch(`${BUDDY_CONFIG.supabaseUrl}/rest/v1/${path}`, {
    headers: {
      apikey: BUDDY_CONFIG.supabaseAnonKey,
      Authorization: `Bearer ${BUDDY_CONFIG.supabaseAnonKey}`,
    },
  });
  if (!r.ok) throw new Error(`supabase ${path} -> ${r.status}`);
  return r.json();
}

/* Pull memory/calendar/todo. Returns the same shape whether from Supabase or
   the empty fallback, so the UI never branches on source. */
async function fetchContext() {
  if (!cfgReady()) return { ...FALLBACK, source: 'local' };
  try {
    const [mem, cal, todo] = await Promise.all([
      sbGet('memory?select=scope,content'),
      sbGet('calendar_events?select=date,title&order=date.asc'),
      sbGet('todo_tasks?select=task,status,ord&order=ord.asc'),
    ]);
    const byScope = Object.fromEntries(mem.map((m) => [m.scope, m.content]));
    return {
      short: byScope.short || '',
      long: byScope.long || '',
      calendar: cal,
      todo: {
        pending: todo.filter((t) => t.status === 'pending').map((t) => t.task),
        completed: todo.filter((t) => t.status === 'completed').map((t) => t.task),
      },
      source: 'supabase',
    };
  } catch (e) {
    console.warn('Supabase fetch failed:', e.message);
    return { ...FALLBACK, source: 'local' };
  }
}

/* POST the student's prompt to /api/execute.

   The API returns exactly four top-level fields - {status, error, response,
   steps} - where `response` is the decision object carrying {action, url,
   message, callback}. The components render a flat shape, so unwrap here and
   leave them alone.

   There is no mock. Everything this returns came from the agent, or is an
   honest report that the agent could not be reached - a fabricated decision
   shown during an outage is worse than no decision at all.

   Two kinds of failure, both surfaced rather than hidden:
     status:"error"  - a real answer. The agent read the prompt and is saying
                       it carried no browsing event to judge.
     transport       - the API is unreachable or returned a non-2xx. */
async function runExecute(prompt) {
  const url = BUDDY_CONFIG.executeUrl;
  if (!url) {
    return { action: 'error', message: 'No executeUrl configured - set window.BUDDY_CONFIG.executeUrl.', steps: [], source: 'remote', isError: true };
  }
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    const body = await r.json();

    if (body.status === 'error' || !body.response) {
      return {
        action: 'error',
        message: body.error || 'The agent returned no decision.',
        steps: body.steps || [],
        source: 'remote',
        isError: true,
      };
    }
    return { ...body.response, steps: body.steps || [], source: 'remote' };
  } catch (e) {
    console.error('execute failed:', e.message);
    return {
      action: 'error',
      message: `Could not reach the agent: ${e.message}`,
      steps: [],
      source: 'remote',
      isError: true,
    };
  }
}


async function apiCloseTab() {
  const url = '/api/close_tab';
  try {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return await r.json();
  } catch (e) {
    console.error('close_tab failed:', e.message);
    return null;
  }
}

async function apiCompleteTask(taskName) {
  const url = '/api/complete_task';
  try {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ task_name: taskName }) });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return await r.json();
  } catch (e) {
    console.error('complete_task failed:', e.message);
    return null;
  }
}

/* favicon-style letter avatar color, derived from domain */
function faviconColor(domain) {
  let h = 0;
  for (let i = 0; i < domain.length; i++) h = (h * 31 + domain.charCodeAt(i)) % 360;
  return `oklch(0.62 0.11 ${h})`;
}
function initial(domain) {
  const core = domain.replace(/^www\./, '').replace(/\.(com|org|co\.il|ac\.il|net|ai|io)$/, '');
  return (core.split('.').pop() || core).charAt(0).toUpperCase();
}

Object.assign(window, {
  BUDDY_CONFIG, fetchContext, runExecute, apiCloseTab, apiCompleteTask, faviconColor, initial,
});
