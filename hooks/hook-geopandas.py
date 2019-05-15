from PyInstaller.utils.hooks import collect_data_files
_osgeo_pyds = collect_data_files('osgeo', include_py_files=True)

osgeo_pyds = []
for p, lib in _osgeo_pyds:
    if '.pyd' in p:
        osgeo_pyds.append((p, ''))

datas = collect_data_files('geopandas', subdir ='datasets')

binaries = osgeo_pyds
