#!/bin/bash

kubectl apply -f MLP_cnn.yml
kubectl cp assignment_2.py cnn-matt:.
kubectl cp a2_mlp.py cnn-matt:.
kubectl cp ./data cnn-matt:.
kubectl exec -it cnn-matt -- /bin/bash << EOF

apt update -y
apt install python3 -y
apt install pip -y
apt install python3.12-venv -y

python3 -m venv ~/nlp
source ~/nlp/bin/activate

pip install yahoo_fin
pip install requests_html
pip install lxml_html_clean
pip install bs4
pip install requests
pip install yfinance
pip install pandas
pip install numpy
pip install transformers
pip install torch

EOF
