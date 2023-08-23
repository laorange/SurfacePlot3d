import datetime
import sys
from pathlib import Path
import os

import numpy as np
import pandas as pd
from loguru import logger
from pyecharts import options as opts
from pyecharts.charts import Surface3D
from scipy.interpolate import griddata

WORK_DIR = Path(sys.argv[0]).parent

# 生成的网格数量
GRID_NUM = 500

# 梯度阈值，如果梯度大于该值，则认为是尖锐的边缘，需要去除
GRADIENT_THRESHOLD = 10


def get_max(array: np.ndarray) -> float:
    """无视NaN，获取numpy数组中的最大值"""
    return float(np.nanmax(array.flatten()))


def get_min(array: np.ndarray) -> float:
    """无视NaN，获取numpy数组中的最小值"""
    return float(np.nanmin(array.flatten()))


class PlotManager:
    def __init__(self, data_path: Path, skip_header=0, skip_footer=0):
        """
        画图控制器，用于读取数据并画图
        :param data_path: 数据文件路径
        :param skip_header: 指定文件头部跳过的行数
        :param skip_footer: 指定文件尾部跳过的行数
        """
        data_path = data_path.resolve()
        self.data_path = data_path

        logger.info(f"正在读取{data_path}，请稍后...")
        self.df: pd.DataFrame = pd.read_csv(
            data_path,
            delim_whitespace=True,
            header=None,
            skip_blank_lines=True,
            skipinitialspace=True,
            names=["x", "y", "z"],
            skiprows=skip_header,
            skipfooter=skip_footer,
            engine="python",
        )

        column_num = self.df.shape[1]
        assert column_num == 3, f"数据列数不为3，而是{column_num}，请检查数据格式"

        logger.info(
            f"共读取到了{column_num}列、{self.df.shape[0]}行数据，数据前10行如下："
            f"\n{self.df.head(10)}"
        )

        self.array = self.df.to_numpy()

        self.x: np.ndarray = self.array[:, 0]
        self.y: np.ndarray = self.array[:, 1]
        self.z: np.ndarray = self.array[:, 2]

    @staticmethod
    def filter_z(z: np.ndarray):
        """Remove outliers and sharp edges from z"""
        # Remove outliers
        Q1 = np.nanpercentile(z, 25)
        Q3 = np.nanpercentile(z, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        z[(z < lower_bound) | (z > upper_bound)] = np.NaN

        # Remove sharp edges
        gradient_zx = np.gradient(z, axis=0)
        gradient_zx[np.isnan(gradient_zx)] = -1e8
        gradient_zy = np.gradient(z, axis=1)
        gradient_zy[np.isnan(gradient_zy)] = -1e8
        z[np.abs(gradient_zx) + np.abs(gradient_zy) > GRADIENT_THRESHOLD] = np.NaN

        return z

    @logger.catch(reraise=True)
    def generate_plot(
        self,
        proportion_of_thickness: float,
        is_show_wire_frame: bool = False,
    ):
        """
        生成3D图
        :param proportion_of_thickness: 用于控制厚度的比例，取值范围为(0, 1)
        :param is_show_wire_frame: 是否显示网格线
        :return:
        """
        OUTPUT_DIR = WORK_DIR / "output"
        if not OUTPUT_DIR.exists():
            OUTPUT_DIR.mkdir()

        logger.info("正在生成网格，请稍后...")
        X, Y = np.meshgrid(
            np.linspace(np.nanmin(self.x), np.nanmax(self.x), GRID_NUM),
            np.linspace(np.nanmin(self.y), np.nanmax(self.y), GRID_NUM),
        )
        Z = griddata((self.x, self.y), self.z, (X, Y), method="cubic")

        # Remove outliers and sharp edges from z
        Z = self.filter_z(Z)

        # make x, y, z start from 0
        X -= np.nanmin(X)
        Y -= np.nanmin(Y)
        Z -= np.nanmin(Z)

        # Prepare data for pyecharts
        data_surface = [
            [X[i, j], Y[i, j], Z[i, j]]
            for i in range(X.shape[0])
            for j in range(X.shape[1])
        ]

        logger.info("正在生成图像，请稍后...")
        # Create a Surface3D chart instance
        surface3d = Surface3D(init_opts=opts.InitOpts(width="1400px", height="800px"))

        # 厚度的占比
        z_height = get_max(Z) - get_min(Z)
        margin = z_height / proportion_of_thickness * (1 - proportion_of_thickness) / 2
        logger.debug(f"厚度为{z_height}(占比为{proportion_of_thickness})，留白高度为{margin}")

        # Add the surface data
        surface3d.add(
            series_name="surface",
            shading="color",
            data=data_surface,
            is_show_wire_frame=is_show_wire_frame,
            xaxis3d_opts=opts.Axis3DOpts(
                type_="value",
                min_=get_min(X),
                max_=get_max(X),
            ),
            yaxis3d_opts=opts.Axis3DOpts(
                type_="value",
                min_=get_min(Y),
                max_=get_max(Y),
            ),
            zaxis3d_opts=opts.Axis3DOpts(
                type_="value",
                min_=get_min(Z) - margin,
                max_=get_max(Z) + margin,
            ),
            grid3d_opts=opts.Grid3DOpts(width=100, height=100, depth=100),
        )

        # Add visual and rendering options
        surface3d.set_global_opts(
            visualmap_opts=opts.VisualMapOpts(
                dimension=2,
                min_=get_min(Z),
                max_=get_max(Z),
                range_color=[
                    "#313695",
                    "#4575b4",
                    "#74add1",
                    "#abd9e9",
                    "#e0f3f8",
                    "#ffffbf",
                    "#fee090",
                    "#fdae61",
                    "#f46d43",
                    "#d73027",
                    "#a50026",
                ],
            )
        )

        # Render the plot
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"{self.data_path.stem}_{datetime_str}.html"
        surface3d.render(str(output_path))
        logger.info(f"结果已生成，路径为：{output_path}")
        os.startfile(output_path)


@logger.catch(reraise=False)
def main():
    while 1:
        try:
            _skip_header = int(input("【?】请输入文件头部跳过的行数："))
            _skip_footer = int(input("【?】请输入文件尾部跳过的行数："))
            _proportion_of_thickness = float(input("【?】请输入厚度的比例（0, 1）："))
            break
        except ValueError:
            logger.info("-" * 10 + "输入的不是数字。请重新输入" + "-" * 10)

    is_show_wire_frame = input("【?】是否显示网格线？（y/n）：").lower() == "y"

    for arg in sys.argv[1:]:
        _path = Path(arg)
        if _path.exists() and _path.is_file():
            PlotManager(
                data_path=_path,
                skip_header=_skip_header,
                skip_footer=_skip_footer,
            ).generate_plot(
                proportion_of_thickness=_proportion_of_thickness,
                is_show_wire_frame=is_show_wire_frame,
            )


if __name__ == "__main__":
    main()
    logger.info("程序结束，按回车键退出：")
    input()
