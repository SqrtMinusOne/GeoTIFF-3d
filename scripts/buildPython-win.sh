#!/usr/bin/env bash
if ! [[ -d venv ]]; then
    cd ..
fi
if [[ -d venv/bin ]]; then
    source venv/bin/activate;
else
    source venv/Scripts/activate;
fi
if [[ -d build ]]; then
    rm -rf build
fi
if [[ -d dist ]]; then
    rm -rf dist
fi

name="geotiff-3d"

pyi_path="$(python -c 'import PyInstaller; print(PyInstaller.__file__[:-11])')"
echo "PyInstaller path: ${pyi_path}"
yes | cp hooks/hook-shapely.py ${pyi_path}hooks


pyqt_path="$(python -c 'import PyQt5.QtCore; print(PyQt5.QtCore.__file__[:-11])')"

paths=(
    src/
    src/api/
    src/shaders/
    src/ui/
    src/ui/widgets/
    src/ui_compiled/
    "C:\Program Files (x86)\Windows Kits\10\Redist\10.0.17763.0\ucrt\DLLs\x86"
)

datas=(
    "src/shaders/shader.frag;shaders/"
    "src/shaders/shader.vert;shaders/"
)

final_path=""
final_datas=""

for i in "${paths[@]}"; do
    final_path="${final_path};${i}"
done

for i in "${datas[@]}"; do
    final_datas="${final_datas} --add-data=${i}"
done

pyinstaller src/main.py --paths="${final_path:1}" --additional-hooks-dir=hooks/ ${final_datas:1} --name=${name} --icon=res/logo-96px.ico
