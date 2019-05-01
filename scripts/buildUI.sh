#!/usr/bin/env bash
if ! [[ -d src ]]; then
    cd ..
fi
source venv/bin/activate;
if ! [[ -d src/ui_compiled ]]; then
    mkdir src/ui_compiled;
fi
rm -rf src/ui_compiled/*;
for filename in ui/*.ui; do
    echo Converting ${filename:3:-3}.py;
    pyuic5 ${filename} --import-from res_compiled >> src/ui_compiled/${filename:3:-3}.py;
done;
