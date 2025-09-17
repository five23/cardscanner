import pathlib
import subprocess

from cardscanner.config import (
    ACTION_NAME, ACTION_SET, PHOTOSHOP_BUNDLE_ID
)

def _esc_js_path(p: pathlib.Path) -> str:
    # Use POSIX-style path; escape backslashes and single quotes for JS string
    s = p.resolve().as_posix()
    return s.replace("\\", "\\\\").replace("'", "\\'")

def run_ps_action(in_path: pathlib.Path, out_path: pathlib.Path, *, quality: int = None) -> pathlib.Path:
    """
    Open a file in Photoshop, run a predefined Action, save result as JPEG.
    Uses AppleScript to invoke Photoshop's ExtendScript (JS) engine.
    """
    quality = JPEG_QUALITY if quality is None else int(quality)

    in_js  = _esc_js_path(in_path)
    out_js = _esc_js_path(out_path)
    out_dir_posix = out_path.parent.resolve().as_posix()

    ascript = f'''
    do shell script "mkdir -p " & quoted form of "{out_dir_posix}"
    tell application id "{PHOTOSHOP_BUNDLE_ID}"
        activate
        set jsOpen to "var f=new File('{in_js}'); if(!f.exists) 'ERROR: missing ' + f.fsName; else{{app.open(f);'OPEN_OK';}}"
        set rOpen to do javascript jsOpen
        if rOpen does not start with "OPEN_OK" then error rOpen

        set jsAct to "app.displayDialogs=DialogModes.NO; try{{app.doAction('{ACTION_NAME}','{ACTION_SET}');'ACT_OK';}}catch(e){{'ERROR: '+e;}}"
        set rAct to do javascript jsAct
        if rAct does not start with "ACT_OK" then error rAct

        set jsSave to "var f=new File('{out_js}'); var o=new JPEGSaveOptions(); o.quality={quality}; app.activeDocument.saveAs(f,o,true);'SAVE_OK';"
        set rSave to do javascript jsSave
        if rSave does not start with "SAVE_OK" then error rSave

        close current document saving no
    end tell
    '''

    try:
        subprocess.run(["osascript", "-e", ascript], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # Bubble up the most helpful message
        msg = e.stderr or e.stdout or str(e)
        raise RuntimeError(f"Photoshop action failed: {msg.strip()}") from e

    return out_path
