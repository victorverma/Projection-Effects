#!/bin/sh

# Downloads version 1.2 of the SWAN-SF dataset into data/swan_sf/. Run this in
# data/.
#
# See this link for information on the dataset:
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM&version=1.2
# See this link for information on downloading data from the Harvard Dataverse:
# https://guides.dataverse.org/en/6.5/api/dataaccess.html

set -e

SERVER_URL="https://dataverse.harvard.edu"
VERSION=1.2
PERSISTENT_ID="doi:10.7910/DVN/EBCFKM"
FORMAT="original"
ZIP_URL="${SERVER_URL}/api/access/dataset/:persistentId/versions/${VERSION}?persistentId=${PERSISTENT_ID}&format=$FORMAT"
ZIP_NAME="swan_sf.zip"
DEST_DIR="swan_sf"

echo "Downloading the ZIP..."
download_failed=0
if command -v curl >/dev/null 2>&1; then
    # With set -e, errors cause an immediate exit, so the second part is needed
    # to print the manual download instructions if there's an error
    curl -fL --retry 5 -o "$ZIP_NAME" "$ZIP_URL" || download_failed=1
elif command -v wget >/dev/null 2>&1; then
    wget -t 5 -O "$ZIP_NAME" "$ZIP_URL" || download_failed=1
else
    echo "ERROR: Need 'curl' or 'wget' to download the ZIP." >&2
    exit 1
fi

if [ $download_failed -ne 0 ] || [ ! -s "$ZIP_NAME" ]; then
  echo "Download failed."
  echo "Please download the ZIP manually from:"
  echo "  https://dataverse.harvard.edu/dataset.xhtml?persistentId=${PERSISTENT_ID}&version=$VERSION"
  echo "Save it as data/${ZIP_NAME} and then unzip it into data/${DEST_DIR}/."
  [ -f "$ZIP_NAME" ] && rm -f "$ZIP_NAME"
  [ -d "$DEST_DIR" ] && rm -rf "$DEST_DIR"
  exit 2
fi

echo "Unzipping..."
rm -rf "$DEST_DIR"
mkdir "$DEST_DIR"
if command -v unzip >/dev/null 2>&1; then
  unzip -q -o "$ZIP_NAME" -d "$DEST_DIR"
elif command -v 7z >/dev/null 2>&1; then
  7z x -y "$ZIP_NAME" -o"$DEST_DIR"
elif command -v bsdtar >/dev/null 2>&1; then
  bsdtar -x -f "$ZIP_NAME" -C "$DEST_DIR"
else
  echo "ERROR: Need 'unzip', '7z', or 'bsdtar' to unzip." >&2
  echo "ZIP saved as $ZIP_NAME" >&2
  exit 3
fi
rm -f "$ZIP_NAME"

echo "Expanding component archives..."
for t in "$DEST_DIR"/*.tar.gz; do
  [ -e "$t" ] || continue
  tar -xz -f "$t" -C "$DEST_DIR"
  rm -f "$t"
done
