# Render the slide HTML files to 1920x1080 PNG with headless Chrome: bash render.sh 1_gioi_thieu
C="/c/Program Files/Google/Chrome/Application/chrome.exe"
for n in "$@"; do
  "$C" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 --window-size=1920,1080 \
       --screenshot="$(cygpath -w "$PWD/$n.png")" "file:///$(cygpath -m "$PWD/$n.html")" 2>/dev/null
done
