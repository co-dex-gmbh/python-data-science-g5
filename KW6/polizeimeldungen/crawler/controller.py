import re
import urllib.parse

import requests
from bs4 import BeautifulSoup
from crawler import URL
from crawler.model import PoliceReport
import json


def curl_page(url):
    raw_page_path = f'raw_pages/{url.replace("/", "_")}.html'
    try:
        with open(raw_page_path, "r") as file:
            body = file.read()
    except FileNotFoundError:
        body = requests.get(url).text
        with open(raw_page_path, "w") as file:
            file.write(body)
    return body



def get_report_basic_data(page):
    url = urllib.parse.urljoin(URL, f"?page_at_1_6={page}")  # Seitenfilter
    body = curl_page(url)
    body = BeautifulSoup(body, "html.parser")

    table = body.find("ul", class_="list--tablelist")

    # Darin finden wir viele Listen-Elemente, die die einzelnen Meldungen enthalten
    items = table.find_all("li")

    police_reports = []
    for item in items:
        police_report = PoliceReport()
        police_report.date = item.find("div", class_="cell nowrap date").text
        police_report.title = item.find_next("a").text
        police_report.link = item.find_next("a")["href"]
        category = item.find("span", class_="category")
        if category:
            police_report.location = category.text.split(": ")[
                1
            ]
        police_reports.append(police_report)
        # print(police_report, end="\n ----- \n")
    return police_reports




def get_report_details(police_reports):
    for report in police_reports:
        detail_url = urllib.parse.urljoin(URL, report.link)
        # print(detail_url)
        report_body = curl_page(detail_url)
        report_body = BeautifulSoup(report_body, "html.parser")

        details = report_body.find("div", class_="textile")
        report_number = details.find_next(string=lambda text: "Nr." in text)
        if report_number:
            report.number = report_number.split(".")[1].lstrip()
        # report.number = (
        #     details.find_next(string=lambda text: "Nr." in text)
        #     .text.split(".")[1]
        #     .lstrip()
        # )  # Wir wollen wieder nur die Nummer # Kann man auch wieder vergessen. Manchmal vergessen die Verfasser, die Nr. dick zu schreiben (beispiel hier https://www.berlin.de/polizei/polizeimeldungen/2024/pressemitteilung.1452880.php). Deswegen filtern wir mit regex

        # TODO: details need to be expanded on all paragraphs!
        report.details = details.find_next("p").text

        # Das hier teilt den String, wenn irgendwo ein Nr. gefolgt von 4 Zahlen steht und entfernt leerzeichen vor dem String.
        report.details = re.split("Nr. [0-9]{4}", report.details)[-1].lstrip()
        # print(re.split("Nr. [0-9]{4}", report.details))
        # print(report)
        # print("-------")
    return police_reports


def get_police_reports():
    # Zusammenbauen
    police_reports = []
    for page in range(1, 4):
        # print(page)
        police_reports += get_report_basic_data(page)

    police_reports = get_report_details(police_reports)
    return police_reports


def safe_reports(police_reports):
    reports_data = [report.__dict__ for report in police_reports]
    with open("police_reports.json", "w") as f:
        json.dump(reports_data, f, ensure_ascii=False, indent=4)