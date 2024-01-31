import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplfinance.original_flavor import Line2D, Rectangle # 调取mplfinance包中的基础函数，用来画线和柱状图
import matplotlib
matplotlib.use('agg') # 关闭matplotlib的交互功能，避免内存泄露
import matplotlib.dates as mdates
import datetime
from PIL import Image
import os

# 相比于mplfinance包的源代码，改变了一个小细节，让蜡烛芯更粗
def new_candlestick(ax, quotes, width=0.2, colorup='k', colordown='r', alpha=1.0, ochl=True):

    OFFSET = width / 2.0

    lines = []
    patches = []
    for q in quotes:
        if ochl:
            t, open, close, high, low = q[:5]
        else:
            t, open, high, low, close = q[:5]

        if close >= open:
            color = colorup
            lower = open
            height = close - open
        else:
            color = colordown
            lower = close
            height = open - close
        
        vline = Line2D(
            xdata=(t, t), ydata=(low, high),
            color=color,
            linewidth=2.0, # 原来是1.0
            antialiased=True,
        )

        rect = Rectangle(
            xy=(t - OFFSET, lower),
            width=width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect.set_alpha(alpha)

        lines.append(vline)
        patches.append(rect)
        ax.add_line(vline)
        ax.add_patch(rect)
    ax.autoscale_view()

    return lines, patches

def new_candlestick_ohlc(ax, quotes, width=0.2, colorup='k', colordown='r', alpha=1.0):
    # 调用自定义的new_candlestick，原来是candlestick
    return new_candlestick(ax, quotes, width=width, colorup=colorup, colordown=colordown, alpha=alpha, ochl=False)

def single_K_Line_Image(open_data, high_data, low_data, close_data, volume_data, stock, date, window=21): # 单日单只股票的蜡烛图
    # 原始数据按照交易日窗口切片
    tradingdays = list(open_data['Date'].astype(str))
    date_index = tradingdays.index(date)
    open_ = list(open_data.loc[:, stock][date_index - window:date_index])
    high = list(high_data.loc[:, stock][date_index - window:date_index])
    low = list(low_data.loc[:, stock][date_index - window:date_index])
    close = list(close_data.loc[:, stock][date_index - window:date_index])
    volume = list(volume_data.loc[:, stock][date_index - window:date_index])
    # 跳过异常数据
    if pd.isnull(np.sum(open_)) or pd.isnull(np.sum(high)) or pd.isnull(np.sum(low)) or pd.isnull(np.sum(close)) or pd.isnull(np.sum(volume)):
        return False
      
    data = pd.DataFrame({'Date':tradingdays[date_index - window:date_index],
                         'Open':open_,
                         'High':high,
                         'Low':low,
                         'Close':close,
                         'Volume':volume})
    data.set_index('Date', inplace=True)
    data = data.astype(float)
    data['Date'] = list(map(lambda x:mdates.date2num(datetime.datetime.strptime(x, '%Y-%m-%d')), data.index.tolist()))

    # 移动平均线
    # data['MA_3'] = data['Close'].rolling(3).mean()
    # data['MA_5'] = data['Close'].rolling(5).mean()
    # data['MA_10'] = data['Close'].rolling(10).mean()

    fig = plt.figure(figsize=(5, 5))  
    grid = plt.GridSpec(5, 5, wspace=0, hspace=0)
    # 成交价数据
    data_price = data[['Date', 'Open', 'High', 'Low', 'Close']]
    data_price = data_price.copy()
    data_price.loc[:, 'Date'] = range(len(data_price))
    # 用成交价数据绘制K线图
    ax1 = fig.add_subplot(grid[0:4, 0:5]) # 设置K线图的尺寸
    ax1.patch.set_facecolor('black') # 设置为黑色背景，其三个通道为均为0
    new_candlestick_ohlc(ax1, data_price.values.tolist(), width=0.9, colorup='red', colordown='green', alpha=1)
    # ax1.plot(list(data['MA_3']), color='orange', linewidth=3)
    # ax1.plot(list(data['MA_5']), color='purple', linewidth=3)
    # ax1.plot(list(data['MA_10']), color='black', linewidth=3)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    # ax1.axis('off')    
    
    # 成交量数据
    data_volume = data[['Date', 'Close', 'Open', 'Volume']]
    data_volume = data_volume.copy()
    data_volume['color'] = data_volume.apply(lambda row: 1 if row['Close'] >= row['Open'] else 0, axis=1)
    data_volume.Date = data_price.Date
    
    # 绘制成交量柱状图
    ax2 = fig.add_subplot(grid[4:5, 0:5])
    ax2.patch.set_facecolor('black')
    # 收盘价高于开盘价为红色，反之为绿色
    ax2.bar(data_volume.query('color==1')['Date'], 
            data_volume.query('color==1')['Volume'], 
            width=0.9, 
            color='red') 
    ax2.bar(data_volume.query('color==0')['Date'], 
            data_volume.query('color==0')['Volume'], 
            width=0.9, 
            color='green')
    # plt.xticks(rotation=30) 
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    # ax2.axis('off')
    # 保存原始图片
    pro_filename = '/Users/liupeilin/Desktop/K_Line_Image/' + date + '/' + stock + '.jpg'
    try:
        pro_file_dir = os.path.dirname(pro_filename)
        if not os.path.isdir(pro_file_dir):
            os.makedirs(pro_file_dir)
    except:
        pass
    fig.savefig(pro_filename, bbox_inches='tight', dpi=60)

    # 读取原始图片，对图片矩阵进行裁剪后再次保存图片
    image_matrix = np.array(plt.imread(pro_filename))
    new_matrix = image_matrix[6:230, 13:237, :]              
    new_image = Image.fromarray(new_matrix)
    new_image.save(pro_filename)

    plt.clf()
    plt.close('all') # 关闭画布，避免占用内存
    # gc.collect() # 强制垃圾收集

def generate_daily_image(open_data, high_data, low_data, close_data, volume_data, date, stock_pool, window):
    for i in stock_pool:
        single_K_Line_Image(open_data, high_data, low_data, close_data, volume_data, i, date, window)

def generate_dates_image(tradingdays, open_data, high_data, low_data, close_data, volume_data, stock_pool, window):
    for date in tradingdays:
        if date >= '2016-07-01' and date <= '2016-12-31':
            generate_daily_image(open_data, high_data, low_data, close_data, volume_data, date, stock_pool, window)
            print(date + ' is finished!')


open_filename = '/Users/liupeilin/Desktop/股票量价数据/adjopen.csv'
high_filename = '/Users/liupeilin/Desktop/股票量价数据/adjhigh.csv'
low_filename = '/Users/liupeilin/Desktop/股票量价数据/adjlow.csv'
close_filename = '/Users/liupeilin/Desktop/股票量价数据/adjclose.csv'
volume_filename = '/Users/liupeilin/Desktop/股票量价数据/adjvolume.csv'
open_data = pd.read_csv(open_filename)
high_data = pd.read_csv(high_filename)
low_data = pd.read_csv(low_filename)
close_data = pd.read_csv(close_filename)
volume_data = pd.read_csv(volume_filename)

tradingdays = list(pd.read_csv('/Users/liupeilin/Desktop/股票量价数据/tradingdays.csv')['Date'])
stock_pool = open_data.columns.tolist()[1:]
window = 21
    
generate_dates_image(tradingdays, open_data, high_data, low_data, close_data, volume_data, stock_pool, window)


