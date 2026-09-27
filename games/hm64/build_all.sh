#!/bin/sh
# Full clean build: generate -> pack -> taint -> site (dev server serves D:/n64work/hm64 on :27641).
set -e
W=D:/n64work/hm64
python -m games.hm64.generate games/hm64/spec $W/clean.patch
python -m games.hm64.pack rom $W/base.z64 $W/clean.patch $W/clean.z64
(echo "clean sha1 $(sha1sum < $W/clean.z64 | cut -c1-40)"; python -m games.hm64.taint_report "$W/rom/Harvest Moon 64 (USA).z64" $W/clean.z64 games/hm64/spec) > $W/taint_final.log 2>&1 || true
tail -1 $W/taint_final.log
cp $W/clean.z64 $W/site/game.z64
