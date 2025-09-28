-- Postgres schema for real-time psychology tutor

create table users (
  user_id uuid primary key default gen_random_uuid(),
  email text unique not null,
  name text,
  created_at timestamptz not null default now(),
  preferences jsonb not null default '{}'::jsonb
);

create table skills (
  skill_id text primary key,
  name text not null,
  parent_skill_id text references skills(skill_id) on delete set null,
  bloom text,
  description text
);

create table skill_prereq (
  skill_id text references skills(skill_id) on delete cascade,
  prereq_skill_id text references skills(skill_id) on delete cascade,
  primary key (skill_id, prereq_skill_id)
);

create table user_skill (
  user_id uuid references users(user_id) on delete cascade,
  skill_id text references skills(skill_id) on delete cascade,
  mastery real not null default 0,
  practice_count int not null default 0,
  success_count int not null default 0,
  last_practiced_at timestamptz,
  current_interval_days int not null default 1,
  next_review_at timestamptz,
  extras jsonb not null default '{}'::jsonb,
  primary key (user_id, skill_id)
);

create table misconceptions (
  tag text primary key,
  skill_id text references skills(skill_id) on delete cascade,
  description text not null
);

create table user_misconception (
  user_id uuid references users(user_id) on delete cascade,
  tag text references misconceptions(tag) on delete cascade,
  evidence real not null default 0,
  updated_at timestamptz not null default now(),
  primary key (user_id, tag)
);

create table question_log (
  question_id uuid primary key default gen_random_uuid(),
  skill_id text references skills(skill_id) on delete set null,
  type text not null check (type in ('mcq','saq')),
  payload jsonb not null,
  created_at timestamptz not null default now()
);

create table user_question_history (
  user_id uuid references users(user_id) on delete cascade,
  question_id uuid references question_log(question_id) on delete set null,
  skill_id text references skills(skill_id) on delete set null,
  presented_at timestamptz not null default now(),
  answered_at timestamptz,
  user_answer jsonb,
  correct boolean,
  score real,
  response_time_ms int,
  misconception_tags text[] default '{}',
  feedback jsonb,
  primary key (user_id, presented_at)
);

create table study_session (
  session_id uuid primary key default gen_random_uuid(),
  user_id uuid references users(user_id) on delete cascade,
  started_at timestamptz not null default now(),
  ended_at timestamptz,
  meta jsonb not null default '{}'::jsonb
);

-- Useful indexes
create index if not exists idx_user_skill_next_review on user_skill (user_id, next_review_at);
create index if not exists idx_uqh_user_time on user_question_history (user_id, presented_at desc);

