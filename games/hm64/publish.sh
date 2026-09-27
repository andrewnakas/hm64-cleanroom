#!/bin/sh
# Publish sources (main) and the web build (gh-pages) to andrewnakas/hm64-cleanroom.
#   games/hm64/publish.sh        (only after taint_report printed "0 failing" into $W/taint_final.log)
# Never publishes the retail ROM, the base image, dirty dumps or dev sites.
set -e
R="$(cd "$(dirname "$0")/../.." && pwd)"
W=D:/n64work/hm64
grep -q " 0 failing" $W/taint_final.log || { echo "refusing: taint_final.log does not say 0 failing"; exit 1; }
[ "$(sha1sum < $W/site/game.z64 | cut -c1-40)" = "$(sha1sum < $W/clean.z64 | cut -c1-40)" ] || { echo "refusing: site ROM is not clean.z64"; exit 1; }
grep -q "$(sha1sum < $W/clean.z64 | cut -c1-40)" $W/taint_final.log || { echo "refusing: taint log is for another ROM"; exit 1; }
EX=$W/export
rm -rf $EX && mkdir -p $EX
cd "$R"
git ls-files cleanroom games/__init__.py games/hm64 ports/emu docs README.md STATUS.md .gitignore | \
  grep -v -e '^games/hm64/portrait_check.py$' -e '^games/hm64/sprite_check.py$' | \
  tar -cf - -T - | (cd $EX && tar -xf -)
cd $EX
git init -q -b main
git -c core.autocrlf=false add -A
git -c user.name=andre -c user.email=treesixtyweather@gmail.com commit -qm "Harvest Moon 64 clean room: generator, spec, tools

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
gh repo view andrewnakas/hm64-cleanroom >/dev/null 2>&1 || gh repo create andrewnakas/hm64-cleanroom --public \
  --description "Harvest Moon 64 in the browser, built from hm64-decomp with every ROM asset regenerated (clean room)"
git remote add origin https://github.com/andrewnakas/hm64-cleanroom.git
git push -q -f origin main
S=$W/site_pub
rm -rf $S && cp -r $W/site $S && cd $S
git init -q -b gh-pages
git add -A
git -c user.name=andre -c user.email=treesixtyweather@gmail.com commit -qm "Web build

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git remote add origin https://github.com/andrewnakas/hm64-cleanroom.git
git push -q -f origin gh-pages
gh api -X POST repos/andrewnakas/hm64-cleanroom/pages -f "source[branch]=gh-pages" -f "source[path]=/" >/dev/null 2>&1 || true
gh api repos/andrewnakas/hm64-cleanroom/pages --jq '.html_url + " " + .status'
