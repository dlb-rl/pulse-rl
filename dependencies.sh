pip config set global.progress_bar off
pip install cryptography google-api-python-client oauth2client==3.0.0
sudo apt-get update && sudo apt-get install -y cmake zlib1g-dev
sudo apt-get update
sudo apt-get install -y build-essential curl unzip psmisc
pip install cython==0.29.0 pytest
sudo apt-get install -y snapd
sudo snap install node --classic --channel=14
sudo apt-get install -y git

git clone https://github.com/dlb-rl/ray.git
ray/ci/travis/install-bazel.sh
cd ray/dashboard/client && npm install && npm run build
cd ../../python && pip install -e . --verbose

cd ../.. && python -m pip install -r requirements.txt