#!/usr/bin/env bash
# Build Bayside for a connected iPhone, install it, and side-load the fine-tuned model.
#   ./deploy-device.sh                 # build + install + copy the model (first time: ~5.5 GB over the cable)
#   SKIP_MODEL=1 ./deploy-device.sh    # app only (the model stays in the app's Documents folder)
#   EDITION=lfm3 ./deploy-device.sh    # the "Bayside LFM3" copy with Liquid LFM3-1.7B (1.4 GB), side by side
# Needs: the iPhone unlocked and connected (USB is fastest), Developer Mode on, and your Apple Developer
# account's latest agreement accepted at developer.apple.com/account.
set -euo pipefail
cd "$(dirname "$0")"
if [ "${EDITION:-gemma}" = "lfm3" ]; then
  SCHEME=BaysideLFM3; BUNDLE=com.gradientharbor.bayside.lfm3
  MODEL=${GH_DEVICE_MODEL:-../use-case-evals-poc/models/ft-lfm3-r2}
else
  SCHEME=Bayside; BUNDLE=com.gradientharbor.bayside
  MODEL=${GH_DEVICE_MODEL:-../use-case-evals-poc/models/ft-e4b-r2}
fi
DEVICE=${DEVICE:-$(xcrun devicectl list devices 2>/dev/null | awk '/physical/ && /iPhone/ && !/unavailable/ {for (i=1;i<=NF;i++) if ($i ~ /^[0-9A-F]{8}-[0-9A-F]{16}$/) {print $i; exit}}')}
[ -n "$DEVICE" ] || { echo "No connected iPhone. Unlock it, connect it (USB or same Wi-Fi), and trust this Mac."; exit 1; }
echo "device: $DEVICE"

(cd HarborKit && ./sync-shared.sh)
(cd Bayside && xcodegen generate >/dev/null)
xcodebuild -project Bayside/Bayside.xcodeproj -scheme "$SCHEME" -configuration Release \
  -destination "id=$DEVICE" -derivedDataPath Bayside/build -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
  -skipMacroValidation build | tail -3
APP=Bayside/build/Build/Products/Release-iphoneos/$SCHEME.app
xcrun devicectl device install app --device "$DEVICE" "$APP"

if [ "${SKIP_MODEL:-0}" != "1" ]; then
  NAME=$(basename "$MODEL")
  echo "copying $NAME ($(du -sh "$MODEL" | cut -f1)) to Documents/Models/$NAME"
  for f in "$MODEL"/*; do
    case "$(basename "$f")" in README.md|LICENSE|*.py|.gitattributes) continue ;; esac
    xcrun devicectl device copy to --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE" \
      --source "$f" --destination "Documents/Models/$NAME/$(basename "$f")" >/dev/null
    echo "  $(basename "$f")"
  done
fi
xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" >/dev/null && echo "launched $SCHEME"
