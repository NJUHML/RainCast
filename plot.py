import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
def plot_precip_image(data, total_hours, leadtime):


    # ===== 固定参数 =====
    levels = [0, 0.1, 1, 5, 10, 20, 40]
    colors = (
        '#FFFFFF',
        '#98FB98',
        '#32CD32',
        '#00BFFF',
        '#0000FF',
        '#FF00FF',
        '#800000'
    )

    lat_start, lat_end = 16.60, 53.35
    lon_start, lon_end = 74.10, 134.85

    figsize = (10, 8)

    # ===== 数据检查与处理 =====
    data = np.asarray(data)

    if data.ndim != 4:
        raise ValueError(
            f"data 必须是 4D，shape 应为 (T, 1, H, W)，当前为 {data.shape}"
        )

    T, C, H, W = data.shape

    if C != 1:
        raise ValueError(
            f"当前函数只支持 C=1 的输入，但 data.shape={data.shape}"
        )

    if T % total_hours != 0:
        raise ValueError(
            f"T={T} 不能被 total_hours={total_hours} 整除，无法做累计"
        )

    if leadtime > T:
        raise ValueError(
            f"leadtime={leadtime} 超出数据时间长度 T={T}"
        )

    if leadtime % total_hours != 0:
        raise ValueError(
            f"leadtime={leadtime} 不能被 total_hours={total_hours} 整除"
        )

    # 去掉通道维度: (T, 1, H, W) -> (T, H, W)
    data = data[:, 0, :, :]

    # 按 total_hours 累计
    # (72, H, W) -> (12, 6, H, W) -> (12, H, W)
    data_acc = data.reshape(
        T // total_hours,
        total_hours,
        H,
        W
    ).sum(axis=1)

    # leadtime=6  -> index 0
    # leadtime=12 -> index 1
    # leadtime=72 -> index 11
    time_idx = leadtime // total_hours - 1

    precip = data_acc[time_idx]

    # ===== 经纬度网格 =====
    lats_1d = np.linspace(lat_start, lat_end, H)
    lons_1d = np.linspace(lon_start, lon_end, W)

    lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)

    # 与你原代码保持一致，翻转纬度
    lats_plot = lats_2d[::-1, :]
    lons_plot = lons_2d[::-1, :]

    # ===== 绘图 =====
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N)

    projection = ccrs.PlateCarree()

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)

    ax.set_extent(
        [lon_start, lon_end, lat_start, lat_end],
        crs=projection
    )

    img = ax.contourf(
        lons_plot,
        lats_plot,
        precip,
        levels=levels,
        colors=colors,
        extend='both',
        transform=projection
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='black')

    xticks = np.arange(80, 135, 15)
    yticks = np.arange(20, 55, 10)

    ax.set_xticks(xticks, crs=projection)
    ax.set_yticks(yticks, crs=projection)

    ax.xaxis.set_major_formatter(LongitudeFormatter(direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter(direction_label=True))

    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)


    cbar = plt.colorbar(
        img,
        ax=ax,
        orientation='vertical',
        pad=0.06,
        shrink=0.85,
        ticks=levels
    )
    cbar.set_label('Precipitation (mm)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.show()

    return




pred = np.load('./output/2024062012.npy')
print(pred.shape)  # (T, C, H, W)
total_hours = 6
leadtime =42
plot_precip_image(pred, total_hours, leadtime)