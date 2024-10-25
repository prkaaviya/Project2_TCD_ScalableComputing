#!/usr/bin/env python3
# Group 36

from requests import get
import os

base_url = "https://cs7ns1.scss.tcd.ie/"
shortname = "paranjik"
# shortname = "padinjav"

filenames = []
with open("paranjik-challenge-filenames.csv") as f:
# with open("padinjav-challenge-filenames.csv") as f:
    for line in f:
        name = line.strip()
        filenames.append(name)
while True:
    downloaded = os.listdir("captchas")
    filenames = list(set(filenames) - set(downloaded))
    
    if len(filenames) == 0:
        break

    print(f"Downloading {len(filenames)} pngs")
    for i, png in enumerate(filenames):
        endpoint = f"{base_url}?shortname={shortname}&myfilename={png}"
        resp = get(endpoint)
        print(f"{i}:\t{png}:\t{resp.status_code}")

        if resp.status_code == 200:
            with open(f"captchas/{png}", "wb") as f:
                f.write(resp.content)