import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import matplotlib.font_manager as fm
import os

# 创建保存图表的目录
if not os.path.exists('covid_visualizations'):
    os.makedirs('covid_visualizations')


# 1. 中文字体设置
def setup_chinese_font():
    """确保图表中文正常显示"""
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    candidate_fonts = ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "WenQuanYi Micro Hei"]

    # 获取系统中安装的字体名称
    available_font_names = []
    for font_path in fm.findSystemFonts():
        try:
            font_name = fm.FontProperties(fname=font_path).get_name()
            available_font_names.append(font_name)
        except:
            continue

    # 查找可用的中文字体
    available_fonts = [f for f in candidate_fonts if f in available_font_names]

    if available_fonts:
        plt.rcParams["font.family"] = available_fonts
        print(f"已启用中文字体: {available_fonts[0]}")
    else:
        print("警告：未检测到中文字体，图表可能显示异常")


# 2. 数据爬取
def crawl_covid_data():
    """从可靠来源爬取COVID-19数据"""
    primary_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    backup_url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"

    for url in [primary_url, backup_url]:
        try:
            print(f"尝试从 {url.split('/')[2]} 获取数据...")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = pd.read_csv(StringIO(response.text))
            print(f"数据获取成功，共 {len(data)} 条记录")
            return data
        except Exception as e:
            print(f"获取失败: {str(e)}，尝试下一个源...")

    print("所有数据源均失败，请检查网络连接")
    return None


# 3. 数据清洗与处理
def clean_and_process(data):
    """处理原始数据，解决类型和分组问题"""
    if data is None:
        return None

    # 识别国家/地区列
    country_col = next((col for col in data.columns if 'country' in col.lower()), None)
    if not country_col:
        country_col = next((col for col in data.columns if 'region' in col.lower()), None)
    if not country_col:
        print("未找到国家/地区相关列")
        return None
    data = data.rename(columns={country_col: 'Country'})
    data['Country'] = data['Country'].astype(str).fillna('Unknown')

    # 识别日期列并转换为数值
    date_cols = [col for col in data.columns if '/' in col or '-' in col]
    for col in date_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    if not date_cols:
        print("未找到有效日期列")
        return None

    # 手动分组
    unique_countries = data['Country'].unique()
    country_data = pd.DataFrame(index=date_cols, columns=unique_countries)

    for country in unique_countries:
        country_rows = data[data['Country'] == country]
        for date_col in date_cols:
            country_data.at[date_col, country] = country_rows[date_col].sum()

    # 转换索引为日期格式
    try:
        country_data.index = pd.to_datetime(country_data.index, format='%m/%d/%y', errors='coerce')
        if country_data.index.isna().sum() > len(country_data) * 0.5:
            country_data.index = pd.to_datetime(country_data.index, format='%Y-%m-%d', errors='coerce')
    except:
        country_data.index = pd.to_datetime(country_data.index, errors='coerce')

    country_data = country_data.dropna()
    return country_data.astype(float)


# 4. 数据分析与可视化
def analyze_and_visualize(country_data):
    """生成分析图表并保存"""
    if country_data is None or country_data.empty:
        return None

    # 计算每日新增病例
    daily_new = country_data.diff().fillna(0)

    # 选择主要国家
    target_countries = ['US', 'India', 'Brazil', 'China', 'Russia', 'United Kingdom']
    valid_countries = [c for c in target_countries if c in daily_new.columns]
    if not valid_countries:
        valid_countries = daily_new.columns[:5].tolist()
    print(f"分析对象: {valid_countries}")

    # 1. 每日新增趋势图
    plt.figure(figsize=(12, 6))
    for country in valid_countries:
        plt.plot(daily_new.index, daily_new[country], label=country, linewidth=1.2)
    plt.title('每日新增确诊病例趋势')
    plt.xlabel('日期')
    plt.ylabel('新增病例数')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('covid_visualizations/daily_trend.png', dpi=300)
    plt.close()
    print("已保存每日趋势图")

    # 2. 累计病例Top10国家
    latest_total = country_data.iloc[-1].sort_values(ascending=False)
    top10 = latest_total.head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top10.values, y=top10.index)
    plt.title('累计病例Top10国家')
    plt.xlabel('累计病例数')
    plt.ylabel('国家')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('covid_visualizations/top10_total.png', dpi=300)
    plt.close()
    print("已保存累计病例Top10图")

    # 返回数据最多的国家用于预测
    return country_data[latest_total.index[0]]


# 5. 优化的预测函数 - 确保符合上升趋势
def predict_future(case_data):
    if case_data is None or len(case_data) < 180:
        print("数据不足180天，无法进行可靠预测")
        return
    country = case_data.name
    print(f"\n开始预测 {country} 未来病例趋势...")

    # 1. 数据准备：确保频率 + 完整时间轴
    series = case_data.asfreq('D')  # 强制每日频率
    all_dates = pd.date_range(start=series.index[0], end=series.index[-1], freq='D')
    # 替换为 ffill() 方法补全缺失日期
    series = series.reindex(all_dates).ffill()

    # 2. 模型训练：用全部历史数据训练
    train = series  # 不再拆分训练/测试，直接用全量数据拟合趋势
    try:
        # SARIMAX 模型：参数适配长期趋势
        model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)

        # 3. 预测：从历史数据最后一天 + 1天开始
        future_days = 30
        # 预测起始日期 = 历史最后一天 + 1天
        forecast_start = series.index[-1] + timedelta(days=1)
        # 生成连续日期：历史最后一天 → 未来30天
        forecast_dates = pd.date_range(start=series.index[-1], periods=future_days + 1, freq='D')[1:]

        # 4. 获取预测结果（带置信区间）
        future_pred = model_fit.get_forecast(steps=future_days)
        future_mean = future_pred.predicted_mean  # 预测均值
        future_ci = future_pred.conf_int()  # 置信区间

        # 5. 绘图：让历史数据和预测数据在时间轴上完全衔接
        plt.figure(figsize=(12, 6))
        # 绘制历史数据（完整时间轴）
        plt.plot(series.index, series, label='历史实际数据', color='blue', linewidth=2)
        # 绘制预测数据（从历史最后一天 +1 开始）
        plt.plot(forecast_dates, future_mean, label='未来30天预测', color='red', linestyle='--', linewidth=2)
        # 绘制置信区间（可选，增强科学性）
        plt.fill_between(forecast_dates,
                         future_ci.iloc[:, 0],
                         future_ci.iloc[:, 1],
                         color='red', alpha=0.1, label='95%置信区间')

        plt.title(f'{country} 未来30天病例预测')
        plt.xlabel('日期')
        plt.ylabel('累计病例数')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'covid_visualizations/{country}_future_pred.png', dpi=300)
        plt.close()

        # 输出未来7天预测
        print("\n未来7天累计病例预测:")
        for i in range(7):
            print(f"{forecast_dates[i].strftime('%Y-%m-%d')}: {int(future_mean.iloc[i])}")

    except Exception as e:
        print(f"预测失败: {str(e)}")


# 主函数
def main():
    print("=== COVID-19数据爬取与分析系统 ===")
    setup_chinese_font()

    # 流程：爬取 -> 处理 -> 分析 -> 预测
    raw_data = crawl_covid_data()
    processed_data = clean_and_process(raw_data)
    target_country_data = analyze_and_visualize(processed_data)
    predict_future(target_country_data)

    print("\n程序完成，图表已保存至 'covid_visualizations' 文件夹")


if __name__ == "__main__":
    main()
