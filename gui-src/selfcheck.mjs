/* Self-check for the two pure helpers in app.jsx: heDir and summarize.

     node gui-src/selfcheck.mjs

   Both are plain functions over plain data, so they are pulled out of the JSX
   source by name and evaluated - no bundler, no DOM, no React. That keeps the
   check honest: it runs the shipped source, not a copy of it that can drift.
*/
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const SRC = join(dirname(fileURLToPath(import.meta.url)), 'app.jsx');
const src = readFileSync(SRC, 'utf8');

/* Slice a top-level `function name(...) {...}` out of the source by brace
   matching. Both targets are brace-balanced and contain no JSX. */
function extract(name) {
  const start = src.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `${name} not found in app.jsx`);
  let i = src.indexOf('{', start), depth = 0;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}' && --depth === 0) return src.slice(start, j + 1);
  }
  throw new Error(`unbalanced braces in ${name}`);
}

const { heDir, summarize } = new Function(
  `${extract('heDir')}\n${extract('summarize')}\nreturn { heDir, summarize };`
)();

/* heDir - direction follows whichever script owns most of the letters. */
assert.equal(heDir('opened reddit.com - r/programming'), 'ltr');
assert.equal(heDir('ממן 16 - מבוא לבינה מלאכותית'), 'rtl');
// The regression: an English sentence citing a Hebrew task name.
assert.equal(
  heDir('ממן 16 is due today and reddit is a high-risk distraction, so I am blocking it.'),
  'ltr',
);
// The mirror case must keep working: a Hebrew sentence citing an English domain.
assert.equal(heDir('פתחת את reddit.com באמצע הלימודים'), 'rtl');
assert.equal(heDir(''), 'ltr');
assert.equal(heDir(null), 'ltr');
assert.equal(heDir('16 :: 2026-08-05'), 'ltr');        // digits are not strong
assert.equal(heDir('אב ab'), 'rtl');                    // tie -> first strong char

/* summarize - the batched tool_call shape must not render as undefined({}). */
const batched = {
  type: 'tool_call',
  tools: [
    { tool: 'read_calendar', args: {} },
    { tool: 'read_todo_list', args: {} },
  ],
};
assert.equal(summarize('response', batched).text, 'read_calendar(), read_todo_list()');
assert.ok(!summarize('response', batched).text.includes('undefined'));

// The single form is untouched.
assert.equal(
  summarize('response', { type: 'tool_call', tool: 'read_website', args: { url: 'reddit.com' } }).text,
  'read_website({"url":"reddit.com"})',
);
assert.equal(summarize('response', { tool: 'read_long_memory', args: {} }).text, 'read_long_memory()');

// A malformed envelope degrades to "?" rather than "undefined".
assert.equal(summarize('response', { type: 'tool_call' }).text, '?()');
assert.equal(summarize('response', { type: 'tool_call', tools: [] }).text, '?()');

// Shapes that share the file must keep their own branches.
assert.equal(
  summarize('response', { type: 'decision', action: 'lock', url: 'reddit.com' }).text,
  'lock reddit.com',
);
assert.equal(summarize('response', { ok: true }).text, 'ok');
assert.equal(summarize('prompt', { messages: [{ role: 'user', content: 'hi' }] }).text,
  '1 messages, last: hi');

console.log('gui selfcheck: all assertions passed');
