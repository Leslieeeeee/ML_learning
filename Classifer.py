def plot_decision_regions(x, y, classifer, resolution = 0.02):
    marker = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'green', 'gray') 
    #从颜色集中选择相应数目的元素
    cmap = ListedColormap(colors[:len(np.unique(colors))])

    #找到输入数据的最大最小值
    x0_min = x[:, 0].min() - 1
    x0_max = x[:, 0].max()
    x1_min = x[:, 1].min() - 1
    x1_max = x[:, 1].max()
    #构建的训练集(分类)
    [x0, x1]= np.meshgrid(np.arange(x0_min, x0_max, resolution),np.arange(x1_min, x1_max, resolution))
    
    z = classifer.sort(np.array([x0.ravel(), x1.ravel()]).T)
    print(z)
    
    #绘图
    z = z.reshape(x0.shape)
    plt.contourf(x0, x1, z, alpha=0.4, cmap = cmap)    
    plt.xlim(x0.min(), x0.max())
    plt.ylim(x1.min(), x1.max())      
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl, 0], y=x[y==cl, 1], alpha=0.8, c=cmap(idx), marker=marker[idx], label=cl)
