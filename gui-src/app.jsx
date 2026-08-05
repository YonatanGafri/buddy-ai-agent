/* Buddy - main interactive app.

   One input, one tab. The textarea is the only way a student talks to Buddy;
   there are no tab-switch, add-URL or close-tab buttons. That is deliberate:
   every one of those was a second path into /api/execute that had to build its
   own prompt, and a grader could reach a decision without ever typing a word.
   Now the prompt the agent judges is exactly the prompt the student wrote.

   What the browser holds is one open tab at a time. It is not seeded and the
   client never picks it - it appears when the agent names a `url` in its
   decision, and carries that decision's action (allow / nudge / lock) as its
   state. The client stores nothing else: a lock applies to the tab now on
   screen, and the next decision replaces it.

   Everything else is display, in three columns - what you tell it, what it
   knows, what it just did: the current tab and the composer, the memory /
   calendar / to-do the agent reads (from Supabase), and the step-by-step trace
   of the run. Plus the final response card and the nudge toast carrying the
   agent's own words.

   `callback` seconds after a reply, the client re-POSTs a wake prompt naming
   the tab still on screen. That is the agent's only follow-up - it is otherwise
   called once per prompt. */

const { useState, useEffect, useRef, useCallback } = React;

/* Rules the generated stylesheet has no equivalent for. Mounted from here
   rather than injected into the HTML by the build script: the bundle's <style>
   is a JSON-escaped blob inside the document, and editing it by string surgery
   is exactly the kind of thing that silently lands in the wrong tag. */
document.head.appendChild(document.createElement('style')).textContent = `
  /* the one tab is a readout, not a control - no hover, no pointer */
  .tablist .row.current { cursor: default; }
  .tab-empty { color: var(--ink-faint); font-size: 14px; line-height: 1.5; padding: 12px; }
  /* the label is the student's whole sentence; clamp it in CSS rather than
     slicing the string, so the full text stays selectable */
  .tablist .row.current .row-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* One tab never fills a column, so the generated flex:1 left a third of the
     window empty above the composer. Both cards size to their contents now and
     the column stacks from the top - the leftover height reads as page
     background instead of as a large empty card. */
  .browser { flex: 0 0 auto; }
  .tablist { flex: 0 0 auto; overflow: visible; }
  .left { justify-content: flex-start; }
  .left .ask { flex: 0 0 auto; }
  .browser-head { padding: 13px 18px 12px; }

  .tab-close {
    background: transparent; border: none; color: var(--ink-faint); font-size: 16px; 
    cursor: pointer; padding: 0 8px; align-self: center; margin-left: auto;
  }
  .tab-close:hover { color: var(--ink); }
  .ctx-item-clickable { cursor: pointer; transition: color 0.15s; }
  .ctx-item-clickable:hover { color: var(--ink); text-decoration: line-through; }
  .ctx-item-clickable:hover .ctx-dot { border-color: var(--ink); background: var(--ink-faint); }

  /* Bigger type throughout. The generated sheet was drawn for a denser layout;
     these are the text elements a person actually reads, nudged up one step. */
  .browser-head h2 { font-size: 17px; }
  .row-domain { font-size: 17px; }
  .row-title { font-size: 14px; }
  .ask-h { font-size: 15px; }
  .ask-box { font-size: 15.5px; line-height: 1.55; }
  .ask-hint { font-size: 12px; }
  .answer-text { font-size: 15px; }
  .answer-prompt { font-size: 13px; }
  .brain-head h3 { font-size: 15px; }
  .ctx-h { font-size: 12px; }
  .ctx-item, .ctx-mem { font-size: 13.5px; }
  .ctx-date { font-size: 12px; flex-basis: 42px; }
  .ctx-tag { font-size: 11px; }
  .trace { font-size: 13px; }
  .nudge-text { font-size: 14.5px; }

  /* Three columns: input, what the agent knows, what it just did. The trace was
     stacked under the context panel in a 560px column, which left it a third of
     the screen height and wrapped nearly every JSON line - it is the widest
     content here and was getting the narrowest box. Its own column fixes that,
     and the two panels stop competing for the same vertical space. */
  .main { grid-template-columns: minmax(340px, 0.9fr) minmax(320px, 1fr) minmax(0, 1.35fr); }
  .ctx { max-height: none; flex: 1 1 auto; border-bottom: 0; }
  /* stacked, not side by side - the column is half as wide as it was */
  .ctx-row { grid-template-columns: 1fr; }
  /* a serialized message array has no spaces to break at, so it overflowed the
     column and put a horizontal scrollbar under the trace. Break anywhere. */
  .trace .msg { min-width: 0; overflow-wrap: anywhere; }
  .brain-log .trace { overflow-x: hidden; }

  /* A summary line is one thought, so it gets one line and an ellipsis rather
     than wrapping to five and pushing the rest of the run off screen. The full
     text is in the raw panel, which is why truncating here loses nothing. */
  /* kind, body and the toggle share one row; only the body is allowed to run
     out of space, so it is the only flex child that shrinks. Making .body a
     block instead stacked all three and tripled every line's height. */
  .logline .msg { display: flex; align-items: baseline; gap: 0; flex-wrap: wrap; }
  /* basis 0, not auto: on auto the body claims its full text width first and
     shoves the raw toggle onto a second line, which is what made every step
     three rows tall. */
  .logline .body { flex: 1 1 0; min-width: 0; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
  .logline.open .body { white-space: normal; }
  .logline .kind, .logline .peek { flex: 0 0 auto; }
  .logline .raw { flex: 1 0 100%; }
  /* prompt / response, so the two halves of a step are told apart at a glance
     without reading either of them. */
  .logline .kind { color: var(--d-faint); font-size: 10px; letter-spacing: 0.04em;
    text-transform: uppercase; margin-right: 6px; }
  .logline .peek { background: none; border: 0; padding: 0 0 0 6px; cursor: pointer;
    color: var(--d-faint); font: inherit; font-size: 10.5px; text-decoration: underline;
    text-underline-offset: 2px; }
  .logline .peek:hover { color: var(--d-ink); }
  .logline .raw { margin: 5px 0 7px; padding: 8px 10px; max-height: 260px;
    overflow: auto; background: oklch(0.22 0.01 250 / 0.55);
    border: 1px solid oklch(1 0 0 / 0.07); border-radius: 6px;
    font-size: 11.5px; line-height: 1.5; color: var(--d-soft);
    white-space: pre-wrap; overflow-wrap: anywhere; }

  /* the endpoint, moved up beside the title - it belongs to what this card does,
     not to the send button, and the footer is now just the button */
  .ask-h { display: flex; align-items: center; gap: 7px; }
  .ask-ep { margin-left: auto; font-size: 11px; font-family: ui-monospace, monospace;
    color: var(--ink-faint); font-weight: 400; letter-spacing: -0.01em; }
  .ask-foot { justify-content: flex-end; }

  /* decision history: newest on top, each card its own row */
  .hist { margin-top: 14px; display: flex; flex-direction: column; gap: 9px; }
  .hist-h { display: flex; align-items: center; gap: 7px; font-size: 12px;
    font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
    color: var(--ink-faint); }
  .hist-n { font-size: 11px; font-weight: 500; letter-spacing: 0; padding: 1px 6px;
    border-radius: 999px; background: var(--line, oklch(0 0 0 / 0.06)); }
  .hist .answer { margin-top: 0; }
  .answer-h { display: flex; align-items: center; gap: 7px; }
  /* the prompt takes the slack so the timestamp sits hard right */
  .answer-h .answer-prompt { flex: 1 1 0; min-width: 0; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
  .answer-t { flex: 0 0 auto; font-size: 11px; color: var(--ink-faint);
    font-variant-numeric: tabular-nums; }
  /* inline favicon: same component as the tab row, sized down */
  .favicon.fav-sm { width: 18px; height: 18px; min-width: 18px; border-radius: 5px;
    font-size: 10px; flex: 0 0 auto; }
  .favicon.fav-sm img { width: 14px; height: 14px; }

  /* brand: mark, name and slogan on one baseline */
  .brand { display: flex; align-items: center; gap: 10px; }
  .brand-mark { display: grid; place-items: center; width: 30px; height: 30px;
    border-radius: 9px; color: #fff; flex: 0 0 auto;
    background: linear-gradient(140deg, oklch(0.62 0.19 264), oklch(0.58 0.18 300));
    box-shadow: 0 1px 3px oklch(0.5 0.15 270 / 0.35); }
  .brand-name { font-size: 20px; font-weight: 650; letter-spacing: -0.02em; }
  .brand-tag { font-size: 12.5px; color: var(--ink-faint); padding-left: 10px;
    border-left: 1px solid var(--line, oklch(0 0 0 / 0.1)); line-height: 1.3; }

  /* on-site clock + the next callback, under the tab title */
  .tab-timers { display: flex; align-items: center; gap: 10px; margin-top: 5px;
    flex-wrap: wrap; }
  .tt { display: inline-flex; align-items: center; gap: 4px; font-size: 11.5px;
    color: var(--ink-faint); font-variant-numeric: tabular-nums; }
  .tt.cb { color: oklch(0.55 0.15 45); font-weight: 550; }
  .tt.cb.due { color: oklch(0.55 0.19 25); }
  .tt.none { opacity: 0.65; }
  /* separator between the two, so they do not read as one sentence */
  .tab-timers .tt + .tt::before { content: '·'; margin-right: 6px; opacity: 0.5;
    font-weight: 400; color: var(--ink-faint); }

  @media (max-width: 900px) { .brand-tag { display: none; } }

  @media (max-width: 1500px) {
    /* not enough width for three: the trace drops full-width beneath the other
       two, which is still wider than the column it came from. The page scrolls
       here rather than each card scrolling inside a fixed viewport height -
       otherwise the to-do list is clipped mid-item with no sign of it. */
    .app { height: auto; min-height: 100vh; }
    .main { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); align-items: start; }
    .brain-ctx .ctx { overflow: visible; }
    .brain-log { grid-column: 1 / -1; min-height: 340px; }
  }
`;

/* The wake prompt, in full. Everything the agent needs to know about why it is
   awake is in its own memory - see the callback comment in runAgent. */
const WAKE = 'Waking up - you asked to check back.';

/* Real wall clock, HH:MM. */
const clock = () => new Date().toTimeString().slice(0, 5);

/* ============ small components ============ */
/* The site's real favicon, via Google's cache. `small` is the inline size used
   in the decision list; the tab row keeps the full one. A domain that has no
   icon - or a fetch that fails offline - falls back to the coloured initial,
   which is why err is tracked rather than left to a broken-image glyph. */
function Favicon({ domain, small }) {
  const [err, setErr] = useState(false);
  const cls = small ? ' fav-sm' : '';
  if (err) {
    return <div className={'favicon' + cls} style={{ background: faviconColor(domain) }}>{initial(domain)}</div>;
  }
  return (
    <div className={'favicon fav-img' + cls}>
      <img src={`https://www.google.com/s2/favicons?domain=${domain}&sz=64`} alt=""
        onError={() => setErr(true)} />
    </div>
  );
}

/* ---- tiny markdown renderer ----
   Covers the subset our memory files use, rendered GitHub-style:
     #/##/### headers, * or - bullet lists, **bold** and `code` inline.
   Not a general parser - just enough to display the .md prose nicely. */
function mdInline(text) {
  // split on **bold** and `code`, keep delimiters
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return parts.map((p, i) => {
    if (p.startsWith('**') && p.endsWith('**')) return <strong key={i}>{p.slice(2, -2)}</strong>;
    if (p.startsWith('`') && p.endsWith('`')) return <code key={i}>{p.slice(1, -1)}</code>;
    return <React.Fragment key={i}>{p}</React.Fragment>;
  });
}

function Markdown({ text }) {
  const lines = (text || '').split('\n');
  const blocks = [];
  let list = null;
  const flush = () => { if (list) { blocks.push(<ul key={'u' + blocks.length}>{list}</ul>); list = null; } };
  lines.forEach((ln, i) => {
    const h = ln.match(/^(#{1,3})\s+(.*)$/);
    const li = ln.match(/^\s*[*-]\s+(.*)$/);
    if (h) { flush(); const T = 'h' + h[1].length; blocks.push(React.createElement(T, { key: i }, mdInline(h[2]))); }
    else if (li) { if (!list) list = []; list.push(<li key={i}>{mdInline(li[1])}</li>); }
    else if (ln.trim()) { flush(); blocks.push(<p key={i}>{mdInline(ln.trim())}</p>); }
    else { flush(); }
  });
  flush();
  return <div className="md">{blocks}</div>;
}

function heDir(s) { return /[֐-׿]/.test(s) ? 'rtl' : 'ltr'; }

/* Drop a leading "# ..." from a memory blob for display only. The row is already
   labelled short / long, so its own title is the same word twice. Display only -
   the agent still reads the file with its heading intact. */
function stripTitle(text) { return (text || '').replace(/^\s*#\s+.*\n?/, ''); }

/* ---- the one open tab ----
   Entirely agent-driven: `domain` is the url the agent said it judged, `label`
   is what the student typed about it, `action` is the decision it carried. No
   click targets - reading it is the point. */
const TAB_COPY = { allow: 'Allowed', nudge: 'Nudged', lock: 'Locked' };

const mmss = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;

/* Two clocks, both re-read from wall time each tick rather than decremented, so
   a backgrounded tab does not drift away from the timer that actually fires.
     on this site  - since the domain first appeared; survives re-decisions of
                     the same tab, resets when a different domain arrives
     next check    - the agent's own callback, counting down. Absent when it set
                     none, which for allow is the normal case and not a fault.
   The callback is the one number that is otherwise invisible until it fires -
   the run says "re-check in 600s" once and then nothing happens for ten
   minutes, which is indistinguishable from a dead timer. */
function TabTimers({ since, due }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);
  const on = Math.max(0, Math.round((Date.now() - since) / 1000));
  const left = due ? Math.round((due - Date.now()) / 1000) : null;
  return (
    <div className="tab-timers">
      <span className="tt"><Icon name="clock" size={11} />{mmss(on)} on this site</span>
      {left !== null ? (
        <span className={'tt cb' + (left <= 0 ? ' due' : '')}>
          {left > 0 ? `next check in ${mmss(left)}` : 'checking now…'}
        </span>
      ) : <span className="tt none">no re-check scheduled</span>}
    </div>
  );
}

function CurrentTab({ tab, busy, onClose }) {
  if (!tab) {
    return (
      <div className="tab-empty">
        {busy ? 'Asking Buddy…' : 'No tab open. Tell Buddy what you opened, below.'}
      </div>
    );
  }
  const cls = 'row current' + (tab.action === 'lock' ? ' blocked' : '');
  return (
    <div className={cls}>
      <Favicon domain={tab.domain} />
      <div className="row-main">
        <div className="row-domain">{tab.domain}</div>
        <div className="row-title" dir={heDir(tab.label)}>{tab.label}</div>
        <TabTimers since={tab.since} due={tab.due} />
      </div>
      <button className="tab-close" onClick={onClose} title="Close tab">✕</button>
      {tab.action === 'lock'
        ? <span className="blocked-pill"><Icon name="lock" size={12} />Blocked</span>
        : <span className={'answer-badge ' + tab.action}>{TAB_COPY[tab.action] || 'Open'}</span>}
    </div>
  );
}

/* ---- context panel: memory + calendar + todo, from the backend ----
   There is no local copy of this data any more, so an unreachable Supabase
   renders as "unreachable" and not as a student with an empty calendar. */
function ContextPanel({ ctx, onCompleteTask }) {
  if (!ctx) {
    return <div className="ctx"><div className="ctx-load">Loading context…</div></div>;
  }
  if (ctx.source !== 'supabase') {
    return (
      <div className="ctx">
        <div className="ctx-load">
          Context unavailable - could not reach the database. The agent reads
          these rows through its own tools, so its decisions are unaffected.
        </div>
      </div>
    );
  }
  return (
    <div className="ctx">
      <div className="ctx-block">
        <div className="ctx-h"><Icon name="brain" size={13} /> Memory</div>
        <div className="ctx-mem"><span className="ctx-tag">short</span><Markdown text={stripTitle(ctx.short)} /></div>
        <div className="ctx-mem"><span className="ctx-tag">long</span><Markdown text={stripTitle(ctx.long)} /></div>
      </div>

      <div className="ctx-row">
        <div className="ctx-block ctx-half">
          <div className="ctx-h"><Icon name="calendar" size={13} /> Calendar</div>
          {ctx.calendar.map((e, i) => (
            <div className="ctx-item" key={i}>
              <span className="ctx-date">{e.date.slice(5)}</span>
              <span className="ctx-txt" dir={heDir(e.title)}>{e.title}</span>
            </div>
          ))}
        </div>

        <div className="ctx-block ctx-half">
          <div className="ctx-h"><Icon name="list" size={13} /> To-do
            <span className="ctx-count">{ctx.todo.pending.length} open</span>
          </div>
          {ctx.todo.pending.map((t, i) => (
            <div className="ctx-item ctx-item-clickable" key={i} onClick={() => onCompleteTask(t)} title="Click to mark as completed"><span className="ctx-dot" />{t}</div>
          ))}
          {ctx.todo.completed.map((t, i) => (
            <div className="ctx-item done" key={'c' + i}><span className="ctx-check">✓</span>{t}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* JSON.stringify escapes the newlines inside string values, so the system
   prompt renders as one unbroken \n\n wall - the thing the raw pane exists to
   let you read. Put the line breaks back, indented to their key's depth. The
   structure is untouched; only the display of the strings changes. */
function prettyJson(v) {
  return JSON.stringify(v, null, 2).replace(
    /"((?:[^"\\]|\\.)*)"/g,
    (m, body) => (body.includes('\\n') ? '"' + body.replace(/\\n/g, '\n') + '"' : m),
  );
}

/* One trace line. Summary by default; click to see the exact object that went
   over the wire, pretty-printed. The raw payload stays reachable because this
   trace is the only window into what the agent was actually sent - a summary
   that cannot be checked against the original is a claim, not evidence. */
function LogLine({ l }) {
  const [open, setOpen] = useState(false);
  const has = !!l.detail;
  return (
    <div className={'logline' + (l.dim ? ' dim' : '') + (open ? ' open' : '')}
         data-tag={l.tag}>
      <span className="t">{l.t}</span>
      <span className="tag">{l.tag}</span>
      <span className="msg">
        {l.kind ? <span className="kind">{l.kind}</span> : null}
        <span className="body" dir={heDir(l.text || '')}>{l.text}</span>
        {has ? (
          <button type="button" className="peek" onClick={() => setOpen(!open)}
                  aria-expanded={open}>{open ? 'hide' : 'raw'}</button>
        ) : null}
        {has && open ? <pre className="raw">{prettyJson(l.detail)}</pre> : null}
      </span>
    </div>
  );
}

function ReasoningLog({ log }) {
  const ref = useRef(null);
  const end = useRef(null);
  /* Follow the tail. Setting scrollTop on .trace alone was not enough: at a
     short viewport the panel fits and an ancestor is the real scroller, so new
     lines landed off screen with nothing moving. Walk up to whichever ancestor
     actually scrolls - but move it only when the last line is genuinely out of
     view. Measured: at 1600x600 the scroller is .main, the whole layout, and
     pinning that unconditionally yanks the calendar out from under whoever is
     reading it. Scroll that ancestor directly rather than via scrollIntoView,
     which would also scroll the window. */
  useEffect(() => {
    const bottom = end.current;
    if (!bottom) return;
    for (let el = bottom.parentElement; el; el = el.parentElement) {
      if (el.scrollHeight <= el.clientHeight + 1) continue;
      const hidden = bottom.getBoundingClientRect().bottom
        > el.getBoundingClientRect().bottom;
      if (hidden) el.scrollTop = el.scrollHeight;
      return;
    }
  }, [log]);
  return (
    <div className="trace" ref={ref}>
      {log.length
        ? log.map((l) => <LogLine key={l.id} l={l} />)
        : <div className="trace-empty">Ask Buddy something to run the agent. The step trace will appear here.</div>}
      <div ref={end} className="trace-end" />
    </div>
  );
}
/* The agent's own words, verbatim. `dir` on the text is the whole RTL story -
   the message is one language, whichever the student wrote in. */
function Nudge({ data, onAct }) {
  if (!data) return null;
  return (
    <div className="nudge-wrap">
      <div className="nudge" key={data.key}>
        <div className="nudge-av"><Icon name="sparkle" size={18} /></div>
        <div className="nudge-body">
          <div className="nudge-who">Buddy · agent</div>
          <div className="nudge-title">A quick nudge</div>
          <div className="nudge-text" dir={heDir(data.message)}>{data.message}</div>
        </div>
        <button className="nudge-x" onClick={() => onAct('dismiss')}><Icon name="x" size={15} /></button>
        <div className="nudge-timer" style={{ animationDuration: '8s' }} />
      </div>
    </div>
  );
}

/* ---- final response card: what the agent answered, in plain language ----
   Spec: "the final response displayed". Never raw JSON. */
const ACTION_COPY = {
  allow: { label: 'Allowed', fallback: 'Nothing to flag - carry on.' },
  nudge: { label: 'Nudge', fallback: 'Buddy sent you a nudge.' },
  lock: { label: 'Locked', fallback: 'Buddy blocked this one for now.' },
};

function AnswerCard({ answer }) {
  if (answer.pending) {
    return (
      <div className="answer pending">
        <div className="answer-h"><span className="answer-badge run">Running</span>
          <span className="answer-prompt">{answer.prompt}</span></div>
        <div className="answer-text">Asking the agent…</div>
      </div>
    );
  }
  const head = (
    <div className="answer-h">
      {answer.url ? <Favicon domain={answer.url} small /> : null}
      <span className={'answer-badge ' + (answer.isError ? 'error' : answer.action)}>
        {answer.isError || answer.action === 'error'
          ? 'Error' : (ACTION_COPY[answer.action] || ACTION_COPY.allow).label}
      </span>
      <span className="answer-prompt" dir={heDir(answer.prompt)}>{answer.prompt}</span>
      <span className="answer-t">{answer.t}</span>
    </div>
  );
  if (answer.isError || answer.action === 'error') {
    return (
      <div className="answer" data-action="error">
        {head}
        <div className="answer-text" dir={heDir(answer.message || '')}>{answer.message}</div>
      </div>
    );
  }
  const copy = ACTION_COPY[answer.action] || ACTION_COPY.allow;
  return (
    <div className="answer" data-action={answer.action}>
      {head}
      <div className="answer-text" dir={heDir(answer.message || '')}>{answer.message || copy.fallback}</div>
      {answer.callback ? (
        <div className="answer-cb"><Icon name="clock" size={12} />
          Re-check in {answer.callback}s
        </div>
      ) : null}
    </div>
  );
}

/* Every decision this session, newest first.
   The card used to be a single slot, so a nudge at 19:47 and the wake that
   escalated it ten minutes later were the same pixel - the second overwrote the
   first and the chain the agent was building could not be read back. Stacking
   them makes the escalation visible as escalation. Kept in state only: a reload
   starts an empty session, same as the trace. */
function DecisionHistory({ answers }) {
  if (!answers.length) return null;
  return (
    <div className="hist">
      <div className="hist-h">Decisions<span className="hist-n">{
        answers.filter((a) => !a.pending).length
      }</span></div>
      {answers.map((a) => <AnswerCard key={a.id} answer={a} />)}
    </div>
  );
}

/* One-line summary of a step's payload, in the reader's terms rather than the
   wire's. The prompt of a ReAct.LLM step is the whole conversation - system
   prompt plus every few-shot example plus the live turns, ~15k characters,
   re-sent verbatim on every turn. Stringified onto one line it buried the
   trace, and pretty-printing it would only make the wall taller. What is
   actually new each turn is the last message and the model's reply, so that is
   what the line shows; `detail` keeps the full object one click away. */
function summarize(kind, v) {
  if (v == null) return { text: '' };
  if (typeof v === 'string') return { text: v };

  // A ReAct.LLM/Summarize prompt: only the tail of the array is new.
  if (Array.isArray(v.messages)) {
    const msgs = v.messages;
    const last = msgs[msgs.length - 1] || {};
    const body = typeof last.content === 'string' ? last.content : JSON.stringify(last.content);
    return { text: `${msgs.length} messages, last: ${body}`, detail: v };
  }
  // A tool call: name and arguments read better than the envelope.
  if (v.tool) {
    const args = v.args && Object.keys(v.args).length ? JSON.stringify(v.args) : '';
    return { text: `${v.tool}(${args})`, detail: v };
  }
  // A decision, the thing the whole run exists to produce.
  if (v.type === 'decision' || v.action) {
    const bits = [v.action, v.url].filter(Boolean).join(' ');
    const cb = v.callback ? ` · re-check ${v.callback}s` : '';
    return { text: `${bits}${cb}${v.message ? ` · "${v.message}"` : ''}`, detail: v };
  }
  if (v.type === 'tool_call') {
    return { text: `${v.tool}(${JSON.stringify(v.args || {})})`, detail: v };
  }
  if (v.type === 'error' || v.error) return { text: v.message || v.error, detail: v };
  if (v.raw) return { text: v.raw, detail: v };          // unparseable model output
  if (v.ok === true) return { text: 'ok', detail: v };   // a write that succeeded
  return { text: JSON.stringify(v), detail: v };
}

/* Normalize a steps[] entry into log lines.
     { module, prompt, response }  - one ReAct step; expands to 2 lines
     { tag, text }                 - a flat line the client added itself */
function stepLines(s) {
  if (!s) return [];
  if (s.module || s.prompt || s.response) {
    const mod = (s.module || 'step').toLowerCase();
    const out = [];
    if (s.prompt) {
      const { text, detail } = summarize('prompt', s.prompt);
      out.push({ tag: mod, kind: 'prompt', text, detail });
    }
    if (s.response) {
      const { text, detail } = summarize('response', s.response);
      out.push({ tag: mod, kind: 'response', text, detail, dim: true });
    }
    if (!out.length) out.push({ tag: mod, text: s.text || '' });
    return out;
  }
  return [{ tag: s.tag || 'step', text: s.text || '', dim: s.dim }];
}

/* ============ App ============ */
function App() {
  const [log, setLog] = useState([]);
  const [nudge, setNudge] = useState(null);
  const [ask, setAsk] = useState('');
  // newest first; index 0 is the run in flight while it is pending
  const [answers, setAnswers] = useState([]);
  const [tab, setTab] = useState(null);   // the one open tab, set by the agent
  const [busy, setBusy] = useState(false);
  const [ctx, setCtx] = useState(null);

  const logSeq = useRef(0);
  const ansSeq = useRef(0);
  const nudgeSeq = useRef(0);
  const nudgeTimer = useRef(null);
  const cbTimer = useRef(null);
  const tabRef = useRef(null);  // the tab on screen right now, for the callback
  tabRef.current = tab;

  /* Append lines to the live log. Each /api/execute step maps to one line.
     Stamped with the real clock - the agent reasons about datetime.now(), so a
     simulated one here would date the trace differently from the reasoning in
     it. */
  const pushLog = useCallback((lines) => {
    setLog((L) => {
      // Spread first: a line carries kind/detail as well as text, and listing
      // the fields by hand here silently dropped them when stepLines grew.
      const stamped = lines.map((ln) => ({
        ...ln,
        id: ++logSeq.current, t: clock(),
        tag: (ln.tag || 'step').toUpperCase(),
      }));
      return [...L, ...stamped].slice(-70);
    });
  }, []);

  const showNudge = useCallback((message) => {
    if (!message) return;
    clearTimeout(nudgeTimer.current);
    setNudge({ message, key: ++nudgeSeq.current });
    nudgeTimer.current = setTimeout(() => setNudge(null), 8200);
  }, []);

  /* Pull memory/calendar/todo once on mount. */
  useEffect(() => { fetchContext().then(setCtx); }, []);

  useEffect(() => () => { clearTimeout(nudgeTimer.current); clearTimeout(cbTimer.current); }, []);

  /* Run the agent: POST the prompt, render the steps and the final response,
     put the tab it named on screen, and schedule the callback it asked for.
       prompt    - goes to /api/execute verbatim; this is the whole input
       opts.wake - true when this run came from a callback, not from the student */
  const runAgent = useCallback((prompt, opts = {}) => {
    const { wake = false } = opts;
    const id = ++ansSeq.current;
    setBusy(true);
    setAnswers((A) => [{ id, pending: true, prompt }, ...A]);
    setNudge(null);
    pushLog([{ tag: wake ? 'wake' : 'input', text: prompt }]);

    return runExecute(prompt).then((res) => {
      if (Array.isArray(res.steps)) pushLog(res.steps.flatMap(stepLines));
      const act = res.action || 'allow';
      const failed = res.isError || act === 'error';
      pushLog([{ tag: failed ? 'error' : act, text: `action=${failed ? 'error' : act}`, dim: true }]);

      // The tab on screen is whatever the agent said it judged. An error named
      // no site, so it leaves the previous tab alone. A wake names no new tab
      // either - it re-decides the one already there, so only the action moves.
      // `since` starts the on-site clock and is carried through every
      // re-decision of the same tab - a nudge does not close the site, so the
      // time spent on it keeps running. `due` is the agent's callback, stamped
      // here from the same value the setTimeout below uses, so the countdown on
      // screen and the timer that fires cannot disagree.
      const due = res.callback ? Date.now() + res.callback * 1000 : null;
      if (!failed) {
        const domain = (res.url || '').trim();
        const prev = tabRef.current;
        const same = prev && (!domain || prev.domain === domain);
        const since = same ? prev.since : Date.now();
        if (wake && prev) {
          setTab({ ...prev, action: act, due });
        } else if (domain) {
          setTab({ domain, label: prompt, action: act, since, due });
        } else if (prev) {
          setTab({ ...prev, action: act, due });
        }
        if (act === 'lock' && domain) pushLog([{ tag: 'lock', text: `Blocked ${domain}.` }]);
      }

      if (act === 'lock' || act === 'nudge') showNudge(res.message);

      // A wake names no url - it re-decides the tab already on screen, so take
      // that one for the icon rather than leaving the row blank.
      const shown = (res.url || '').trim() || (tabRef.current || {}).domain || '';
      setAnswers((A) => A.map((a) => (a.id === id ? {
        id, t: clock(), url: failed ? '' : shown,
        action: act, message: res.message, prompt, isError: !!res.isError,
        callback: res.callback || null,
      } : a)));
      setBusy(false);

      // The agent may have called rewrite_memory mid-run, and it writes straight
      // to Supabase - the panel above is holding whatever we read on mount, so
      // it is now stale. Re-read it. The write always lands before the decision
      // (the run ends at the decision), so by here the row is already current.
      fetchContext().then(setCtx);

      // Honor the callback: re-POST after the delay. The wake prompt says only
      // that the timer fired - the client does not tell the agent what it was
      // checking, because the client does not know. The agent wrote that to its
      // own memory before deciding, and reads it back on this turn. Any summary
      // composed here would be the browser's account of the agent's intent.
      clearTimeout(cbTimer.current);
      if (res.callback) {
        pushLog([{ tag: 'callback', text: `re-check in ${res.callback}s`, dim: true }]);
        cbTimer.current = setTimeout(
          () => runAgent(WAKE, { wake: true }),
          res.callback * 1000,
        );
      }
      return res;
    });
  }, [pushLog, showNudge]);


  const closeTab = useCallback(() => {
    if (!tabRef.current) return;
    setTab(null);
    clearTimeout(cbTimer.current);
    apiCloseTab().then(() => fetchContext().then(setCtx));
  }, []);

  const completeTask = useCallback((taskName) => {
    apiCompleteTask(taskName).then(() => fetchContext().then(setCtx));
  }, []);

  /* The only input surface there is. */
  const runAsk = useCallback(() => {
    const text = ask.trim();
    if (!text || busy) return;
    runAgent(text);
  }, [ask, busy, runAgent]);

  const onNudgeAct = useCallback((act) => {
    if (act === 'dismiss') setNudge(null);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Icon name="sparkle" size={17} /></div>
          <div className="brand-name">Buddy</div>
          {/* one line, because it is a subtitle and not a paragraph: what it
              watches, what it does about it, and that a person still decides */}
          <div className="brand-tag">
            watches what you open, nudges you back to the deadline that is closest
          </div>
        </div>
        <div className="spacer"></div>
      </header>

      <main className="main">
        <div className="left">
          <section className="browser">
            <div className="browser-head">
              <h2>Current tab</h2>
            </div>
            <div className="tablist">
              <CurrentTab tab={tab} busy={busy && !tab} onClose={closeTab} />
            </div>
          </section>

          <section className="ask">
            <div className="ask-h"><Icon name="sparkle" size={14} /> Ask Buddy
              <span className="ask-ep">POST /api/execute</span>
            </div>
            <textarea className="ask-box" value={ask} rows={3}
              placeholder="Tell Buddy what you opened - e.g. Opened youtube.com - 'lo-fi beats to study to'"
              onChange={(e) => setAsk(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runAsk(); } }} />
            <div className="ask-foot">
              <button className="run-btn" disabled={!ask.trim() || busy} onClick={runAsk}>
                <Icon name="bolt" size={16} />{busy ? 'Running…' : 'Run Agent'}
              </button>
            </div>
            <div style={{textAlign: 'right', marginTop: '4px'}}>
              <button style={{background: 'transparent', border: 'none', color: 'var(--ink-faint)', fontSize: '11px', cursor: 'pointer'}} disabled={busy} onClick={() => runAgent('Waking up - you asked to check back.', { wake: true })}>
                Wake Up (For Testing)
              </button>
            </div>
            <DecisionHistory answers={answers} />
          </section>
        </div>

        <aside className="brain brain-ctx">
          <div className="brain-head">
            <div>
              <h3>Agent context</h3>
            </div>
          </div>
          <ContextPanel ctx={ctx} onCompleteTask={completeTask} />
        </aside>

        <aside className="brain brain-log">
          <div className="brain-head">
            {/* the pill says "live" already; a heading next to it said it twice */}
            <div className="live"><span className="pulse"></span>LIVE</div>
          </div>
          <ReasoningLog log={log} />
        </aside>
      </main>

      <Nudge data={nudge} onAct={onNudgeAct} />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
