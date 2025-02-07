from crawler.controller import get_police_reports, safe_reports

police_reports = get_police_reports()

print("finished")
safe_reports(police_reports)
# print(len(police_reports))
# for report in police_reports[0:10]:
#     print(report)
