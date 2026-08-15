#!/bin/bash
# package_app.sh — bundle the debug build into a double-clickable FinchmoeChat.app
set -e
cd "$(dirname "$0")"

swift build

APP="FinchmoeChat.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp .build/debug/FinchmoeChat "$APP/Contents/MacOS/FinchmoeChat"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>FinchmoeChat</string>
    <key>CFBundleDisplayName</key><string>FinchMoE Chat</string>
    <key>CFBundleIdentifier</key><string>com.haihengh.finchmoe.chat</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>CFBundleExecutable</key><string>FinchmoeChat</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
</dict>
</plist>
PLIST

echo "Built $APP — drag it to /Applications or double-click to launch."
echo "Note: unsigned ad-hoc — right-click → Open on first launch (or run:"
echo "  xattr -dr com.apple.quarantine $APP"
