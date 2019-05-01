IF NOT EXIST src (
    cd ..
)
call venv\Scripts\activate
IF EXIST src\ui_compiled (
    rm -rf src\ui_compiled\*
) ELSE (
    md src\ui_compiled
)
FOR %%I in (ui\*.*) DO (
	echo Converting %%I
	call pyuic5 %%I --import_from res_compiled >> src\ui_compiled\%%~nI.py
)
echo Finished
