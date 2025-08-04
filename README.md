covid-19-prediction
This is a prediction model for the number of COVID-19 infections using Python.
For the first use, you need to use pip install -r requirements.txt to install the required Python packages.




Python Program Function Analysis Report: COVID-19 Infection Number Prediction System
I. Program Overview
This Python program is a comprehensive tool for crawling, processing, analyzing, and predicting the number of COVID-19 infections. It achieves its goals through the following four main stages: data crawling, data cleaning and processing, data analysis and visualization, and future prediction based on the SARIMAX model.
II. Detailed Explanation of Program Modules
Chinese Font Setting

Function Name: setup_chinese_font()
Function: Ensure that Chinese characters can be correctly displayed in the generated charts. By searching for available Chinese fonts in the system (such as SimHei, Microsoft YaHei, etc.) and setting them as the default font. If no suitable font is found, a warning will be issued.
Quoted Code Snippet:

def setup_chinese_font():
    ...
    plt.rcParams["font.family"] = available_fonts[0]
    ...

Data Crawling

Function Name: crawl_covid_data()
Function: Obtain the latest global COVID-19 confirmed data from two preset data sources (the primary source and the backup source). Use the requests library to send HTTP requests and convert the returned data into a Pandas DataFrame format. If the primary source is unavailable, try the backup source.
Quoted Code Snippet:

def crawl_covid_data():
    ...
    response = requests.get(url, timeout=15)
    data = pd.read_csv(StringIO(response.text))
    ...

Data Cleaning and Processing

Function Name: clean_and_process(data)
Function: Perform necessary cleaning and transformation operations on the raw data, including:

Identifying and renaming the country/region column.
Converting the date column to a numerical type.
Summarizing the number of confirmed cases by date for each country.
Converting the date index to the datetime format.


Quoted Code Snippet:

def clean_and_process(data):
    ...
    country_data.index = pd.to_datetime(country_data.index, format='%m/%d/%y', errors='coerce')
    ...

Data Analysis and Visualization

Function Name: analyze_and_visualize(country_data)
Function: Generate two main analysis charts:

Daily new confirmed cases trend chart: Show the changing trend of daily new cases in the selected countries.
Bar chart of the top 10 countries with the most cumulative cases: Display the top 10 countries with the most cumulative confirmed cases.


Quoted Code Snippet:

def analyze_and_visualize(country_data):
    ...
    plt.plot(daily_new.index, daily_new[country], label=country, linewidth=1.2)
    ...
    sns.barplot(x=top10.values, y=top10.index)
    ...

Optimized Prediction Function

Function Name: predict_future(case_data)
Function: Use the SARIMAX model to predict the development trend of the epidemic in the next 30 days. The specific steps are as follows:

Prepare the data: Ensure that the time series data has a complete daily frequency.
Train the model: Train the SARIMAX model using the entire historical data set.
Predict the future: Starting from the last day of the historical data, predict the cumulative number of cases in the next 30 days and draw a chart of the prediction results.


Quoted Code Snippet:

def predict_future(case_data):
    ...
    model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
    ...
    plt.plot(forecast_dates, future_mean, label='Prediction for the next 30 days', color='red', linestyle='--', linewidth=2)
    ...

III. Main Process Execution

Main Function: main()
Function: Coordinate the entire program's operation process, calling each of the above functional modules in turn. First, set the Chinese font, then perform data crawling, data processing, data analysis and visualization in sequence, and finally make a future prediction. All generated charts will be saved to the specified folder.
Quoted Code Snippet:

def main():
    ...
    raw_data = crawl_covid_data()
    processed_data = clean_and_process(raw_data)
    target_country_data = analyze_and_visualize(processed_data)
    predict_future(target_country_data)
    ...

IV. Summary
This program provides a complete solution, covering the entire process from data acquisition to prediction analysis. It can not only help users understand the current global epidemic development situation but also make scientific and reasonable predictions about the epidemic trend in the future. By reasonably utilizing multiple popular libraries in Python (such as Pandas, Matplotlib, Seaborn, etc.), it ensures the efficiency and accuracy of data processing and visualization.
