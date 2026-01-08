import matplotlib.pyplot as plt

# 绘制 2D 函数图像
def draw_2d_func_pic(x_values, y_values, x_label=None, y_label=None, title=None):
    plt.figure()
    plt.plot(x_values, y_values)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


