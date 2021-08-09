mkdir -p ./inputs/religious_art
# train images
echo "----------------train画像をダウンロードします----------------"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1jWkgtvrD8jv4AEVysYTJC7TmmPq1Rd_4" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1jWkgtvrD8jv4AEVysYTJC7TmmPq1Rd_4" -o ./inputs/religious_art/christ-train-imgs.npz
# train labels
echo "----------------trainラベルをダウンロードします----------------"
curl -L "https://drive.google.com/uc?export=download&id=123TBDokMFZMhZ9QRCt-N5TLiYmCvGHQf" -o ./inputs/religious_art/christ-train-labels.npz
# test images
echo "----------------test画像をダウンロードします----------------"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Pg39HSQrJwu1moe2zlfbYiy2uANex7f2" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Pg39HSQrJwu1moe2zlfbYiy2uANex7f2" -o ./inputs/religious_art/christ-test-imgs.npz