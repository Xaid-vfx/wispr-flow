#!/usr/bin/env bash
# Build Dictation.app via py2app and wrap it in a DMG via hdiutil.
# Produces: dist/Dictation-<version>.dmg

set -euo pipefail

cd "$(dirname "$0")"

VENV_PY=".venv/bin/python"
APP_NAME="Dictation"
VERSION="0.2.0"
DMG_NAME="${APP_NAME}-${VERSION}.dmg"
STAGING_DIR="dist/dmg-staging"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: $VENV_PY not found. Activate or create the venv first." >&2
  exit 1
fi

echo "[1/4] Cleaning previous build artifacts..."
rm -rf build dist

echo "[2/4] Building ${APP_NAME}.app via py2app (this takes a few minutes)..."
if ! "$VENV_PY" setup.py py2app; then
  echo "Error: py2app failed. See output above." >&2
  exit 1
fi

if [[ ! -d "dist/${APP_NAME}.app" ]]; then
  echo "Error: dist/${APP_NAME}.app was not produced — check py2app output above." >&2
  exit 1
fi

# py2app's legacy imp.find_module can't resolve PEP 420 namespace packages
# (no top-level __init__.py), and it zips data-only directories where a
# bundled .dylib would be unreachable by dlopen. Copy these trees into the
# bundle's unzipped site-packages so imports resolve to real filesystem paths.
echo "[2b/4] Copying namespace / data packages into bundle..."
PY_VER="$("$VENV_PY" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
SITE="$("$VENV_PY" -c 'import site, sys; print([p for p in site.getsitepackages() if "site-packages" in p][0])')"
RES_LIB="dist/${APP_NAME}.app/Contents/Resources/lib/${PY_VER}"

NAMESPACE_PKGS=(
  "mlx"                  # namespace pkg; holds core.so + Metal shaders in lib/ & share/
  "_sounddevice_data"    # holds libportaudio.dylib — dlopen can't load from zip
  "tiktoken_ext"         # tiktoken plugin namespace used by mlx_whisper tokenizer
  "PyObjCTools"          # pyobjc namespace utilities used by rumps / AppKit code
)

mkdir -p "$RES_LIB"
for pkg in "${NAMESPACE_PKGS[@]}"; do
  src="${SITE}/${pkg}"
  if [[ -d "$src" ]]; then
    echo "  + $pkg"
    rm -rf "$RES_LIB/$pkg"
    cp -R "$src" "$RES_LIB/"
  else
    echo "  - $pkg (not installed, skipping)"
  fi
done

# py2app also wrote stub/.pyc copies of these packages into the site-packages
# zip. PEP 420 merges __path__ across all sys.path hits with the zip winning,
# which shadows our real filesystem copies and sends dlopen into the zip.
# Strip the zipped copies so imports resolve unambiguously to the filesystem.
PY_NODOT="python$("$VENV_PY" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
ZIP="dist/${APP_NAME}.app/Contents/Resources/lib/${PY_NODOT}.zip"
if [[ -f "$ZIP" ]]; then
  for pkg in "${NAMESPACE_PKGS[@]}"; do
    zip -q -d "$ZIP" "${pkg}/*" 2>/dev/null || true
  done
fi

# Ad-hoc code signing: gives the bundle a stable identity so macOS TCC
# (Input Monitoring, Accessibility, Microphone) can track it reliably
# across rebuilds. Without this, unsigned bundles often show permissions
# as "on" in Privacy & Security while TCC silently rejects them at runtime.
echo "[2c/4] Ad-hoc code signing the bundle..."
codesign --force --deep --sign - "dist/${APP_NAME}.app" >/dev/null 2>&1 || {
  echo "  warning: codesign failed — permissions may drift on rebuild" >&2
}

echo "[3/4] Staging DMG contents..."
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
cp -R "dist/${APP_NAME}.app" "$STAGING_DIR/"
# Drag-to-Applications convenience shortcut
ln -s /Applications "$STAGING_DIR/Applications"

echo "[4/4] Creating ${DMG_NAME}..."
rm -f "dist/${DMG_NAME}"
hdiutil create \
  -volname "${APP_NAME} ${VERSION}" \
  -srcfolder "$STAGING_DIR" \
  -ov \
  -format UDZO \
  "dist/${DMG_NAME}" >/dev/null

rm -rf "$STAGING_DIR"

echo
echo "Done: dist/${DMG_NAME}"
echo "     $(du -h "dist/${DMG_NAME}" | cut -f1) on disk"
