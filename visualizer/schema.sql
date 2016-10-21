drop table if exists entries;
create table entries (
  id integer primary key autoincrement,
  title text not null,
  'text' text not null
);

drop table if exists files;
create table files (
  id integer primary key autoincrement,
  name text not null,
  upload_date text not null,
  content blob not null
);
