-- Buddy - Supabase schema.
-- Three tables. Memory is the only one the agent writes; calendar + todo are
-- read-only context (MCP stand-ins). RLS allows anon SELECT so the GUI can read
-- directly via the REST API; writes go through the service key (server side).

-- ---- memory: two rows, scope = 'short' | 'long' ----
create table if not exists memory (
  scope       text primary key check (scope in ('short', 'long')),
  content     text not null,
  updated_at  timestamptz not null default now()
);

-- ---- calendar_events: read-only context ----
create table if not exists calendar_events (
  id     bigint generated always as identity primary key,
  date   date not null,
  title  text not null
);

-- ---- todo_tasks: read-only context, status = 'pending' | 'completed' ----
create table if not exists todo_tasks (
  id      bigint generated always as identity primary key,
  task    text not null,
  status  text not null check (status in ('pending', 'completed')),
  ord     int  not null default 0
);

-- ---- RLS: anon may read everything, nobody writes via anon ----
alter table memory          enable row level security;
alter table calendar_events enable row level security;
alter table todo_tasks      enable row level security;

drop policy if exists anon_read_memory   on memory;
drop policy if exists anon_read_calendar on calendar_events;
drop policy if exists anon_read_todo     on todo_tasks;

create policy anon_read_memory   on memory          for select to anon using (true);
create policy anon_read_calendar on calendar_events for select to anon using (true);
create policy anon_read_todo     on todo_tasks      for select to anon using (true);
