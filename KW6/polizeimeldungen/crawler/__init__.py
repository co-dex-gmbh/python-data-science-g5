from datetime import datetime


URL = "https://www.berlin.de/polizei/polizeimeldungen/"



# # format the string 15.08.2024 14:56 Uhr to a datetime object
# def format_date(date):
#     # date = date.split(" ")
#     # date = f"{date[0]} {date[1]}"
#     date = datetime.strptime(date, "%d.%m.%Y %H:%M Uhr")
#     return date

# print(format_date("15.08.2024 14:56 Uhr"))