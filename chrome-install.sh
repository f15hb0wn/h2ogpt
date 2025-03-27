mkdir -p /etc/apt/keyrings/
rm -rf /tmp/google.pub
wget https://dl-ssl.google.com/linux/linux_signing_key.pub -O /tmp/google.pub
gpg --no-default-keyring --keyring /etc/apt/keyrings/google-chrome.gpg --import /tmp/google.pub
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' | tee /etc/apt/sources.list.d/google-chrome.list
apt-get update -y

apt-get install google-chrome-stable -y
chromeVersion="$(echo $(google-chrome --version) | cut -d' ' -f3)"
# visit https://googlechromelabs.github.io/chrome-for-testing/ and download matching version
# E.g.
# Attempt to download matching version of ChromeDriver
rm -rf chromedriver-linux64.zip chromedriver LICENSE.chromedriver
if ! wget -O chromedriver-linux64.zip "https://storage.googleapis.com/chrome-for-testing-public/${chromeVersion}/linux64/chromedriver-linux64.zip"; then
    echo "Failed to download ChromeDriver for version ${chromeVersion}, attempting to download known working version 124.0.6367.91."
    if ! wget -O chromedriver-linux64.zip "https://storage.googleapis.com/chrome-for-testing-public/124.0.6367.91/linux64/chromedriver-linux64.zip"; then
        echo "Failed to download fallback ChromeDriver version 124.0.6367.91."
        exit 1
    fi
fi

unzip -o chromedriver-linux64.zip
chown root:root /usr/bin/chromedriver
chmod +x /usr/bin/chromedriver
