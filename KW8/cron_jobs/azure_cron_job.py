import logging
import azure.functions as func
import requests

app = func.FunctionApp()


@app.function_name(name="webiste_poller")
@app.timer_trigger(schedule="0 */5 * * * *",
                   arg_name="webiste_poller",
                   run_on_startup=True)
def fetch_website_content() -> None:
    # Rufe die Webseite auf und speichere den Inhalt
    url = "https://datapluseducation.de"
    response = requests.get(url)
    if response.status_code == 200:
        logging.info(f"Successfully fetched {url}")
    else:
        logging.error(f"Failed to fetch {url}: {response.status_code}")
