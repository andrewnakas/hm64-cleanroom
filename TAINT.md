# Taint scan evidence

Verbatim output of the last taint scan saved in the local work directory, committed as evidence.
It was not re-run for this commit; check the log date against the date of the published build.

Command: `python -m games.hm64.taint_report "<rom>" clean.z64 games/hm64/spec`

Live-build check (2026-10-19): the published `game.z64` (git blob d72e184462) equals `D:/n64work/hm64/clean.z64`, sha1 1755cad9dabe, the ROM named in this log.

## `taint_final.log` (written 2026-09-27 03:46)

```
clean sha1 1755cad9dabe2280358f1e05a57c0743881515d8
taint: 877 slots with short coincidental matches; 0 failing (run >= 32 B) {'tex': 0, 'rgba': 0, 'pal': 0, 'font': 0, 'adpcm': 0, 'pcm': 0}
```
