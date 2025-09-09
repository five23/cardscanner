# run_ps_action_mac.py — open & save via Photoshop JSX (no AppleScript file coercion)
import subprocess, pathlib

PHOTOSHOP_BUNDLE_ID = "com.adobe.Photoshop"   # version-agnostic
ACTION_SET          = "Default Actions"
ACTION_NAME         = "Calibrate Scanned Image"

def run_ps_action(in_path: pathlib.Path, out_path: pathlib.Path, timeout_sec: int = 120):
    in_posix = in_path.resolve().as_posix()
    out_posix = out_path.resolve().as_posix()
    out_dir = out_path.parent.resolve().as_posix()

    ascript = f'''
    do shell script "mkdir -p " & quoted form of "{out_dir}"

    with timeout of {timeout_sec} seconds
        tell application id "{PHOTOSHOP_BUNDLE_ID}"
            activate
            -- absolutely no popups
            do javascript "app.displayDialogs = DialogModes.NO;"

            -- OPEN via JSX (bypasses alias/UTI quirks)
            set jsOpen to "var f=new File('{in_posix}'); if(!f.exists) 'ERROR: missing ' + f.fsName; else{{ app.open(f); 'OPEN_OK'; }}"
            set rOpen to do javascript jsOpen
            if rOpen does not start with "OPEN_OK" then error rOpen

            -- Run your action (must NOT contain interactive steps)
            set jsAct to "try{{ app.doAction('{ACTION_NAME}','{ACTION_SET}'); 'ACT_OK'; }} catch(e){{ 'ERROR: ' + e; }}"
            set rAct to do javascript jsAct
            if rAct does not start with "ACT_OK" then error rAct

            -- SAVE (explicit options → no dialogs)
            set jsSave to "var f=new File('{out_posix}'); var o=new JPEGSaveOptions(); o.quality={JPEG_QUALITY}; app.activeDocument.saveAs(f,o,true); 'SAVE_OK';"
            set rSave to do javascript jsSave
            if rSave does not start with "SAVE_OK" then error rSave

            -- Close without prompts
            close current document saving no
        end tell
    end timeout
    '''

    try:
        subprocess.run(["osascript", "-e", ascript], check=True, timeout=timeout_sec + 10)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Photoshop action call timed out. Your action likely shows a dialog. "
            "Edit the action to remove interactive steps and enable ‘Suppress Color Profile Warnings’ "
            "in Actions → Playback Options."
        )


# Example usage:
# run_ps_action(pathlib.Path("/Users/you/Desktop/input.jpg"),
#               pathlib.Path("/private/tmp/output_PS.jpg"),
#               run_action=True, quality=10)

