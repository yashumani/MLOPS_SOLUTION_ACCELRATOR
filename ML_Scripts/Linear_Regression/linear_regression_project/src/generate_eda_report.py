import os
import sweetviz as sv
from ydata_profiling import ProfileReport
import dtale
from config import config

def generate_eda_report(df, title="EDA"):
    eda_tool = config.get('eda_tool', 'sweetviz')
    if eda_tool == 'ydata-profiling':
        profile = ProfileReport(df, title=title)
        profile.to_file(os.path.join(config['reports_path'], f"{title}_report.html"))
    elif eda_tool == 'dtale':
        dtale.show(df).open_browser()
    elif eda_tool == 'sweetviz':
        report = sv.analyze([df, "Data"])
        report.show_html(os.path.join(config['reports_path'], f"{title}_report.html"), open_browser=False)