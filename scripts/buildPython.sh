#!/usr/bin/env bash
if ! [[ -d venv ]]; then
    cd ..
fi
source venv/bin/activate;
if [[ -d build ]]; then
    rm -rf build
fi
if [[ -d dist ]]; then
    rm -rf dist
fi

name="geotiff-3d"

pyqt_path="$(python -c 'import PyQt5.QtCore; print(PyQt5.QtCore.__file__[:-10])')"

paths=(
    src/
    src/api/
    src/shaders/
    src/ui/
    src/ui/widgets/
    src/ui_compiled/
)

datas=(
    src/shaders/shader.frag:shaders/
    src/shaders/shader.vert:shaders/
    ${pyqt_path}/Qt/plugins/xcbglintegrations/libqxcb-egl-integration.so:PyQt5/Qt/plugins/xcbglintegrations/
    ${pyqt_path}/Qt/plugins/xcbglintegrations/libqxcb-glx-integration.so:PyQt5/Qt/plugins/xcbglintegrations/
)

final_path=""
final_datas=""

for i in "${paths[@]}"; do
    final_path=${final_path}:${i}
done

for i in "${datas[@]}"; do
    final_datas="${final_datas} --add-data=${i}"
done
# echo "src/main.py --paths=${final_path:1} --additional-hooks-dir=hooks/ ${final_datas:1}"
pyinstaller src/main.py --paths=${final_path:1} --additional-hooks-dir=hooks/ ${final_datas:1} --name=${name} --onefile

# cp -r venv/lib/python3.6/site-packages/PyQt5/Qt/plugins/xcbglintegrations dist/main/PyQt5/Qt/plugins/
