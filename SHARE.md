# 如何分享 dlfc-code

## 对方需要什么环境
- Python 3.10+（任意系统），安装 numpy、matplotlib：
  `pip install numpy matplotlib`
- 无需 PyTorch/TensorFlow —— 全部纯 numpy 从零实现
- MNIST 首次运行会自动下载（Ch10/Ch11），无需手动准备数据

## 方式一：直接压缩包（最简单）
1. 把 `dlfc-code` 文件夹打包成 zip（可含 .git，也可删掉只保留代码）；
2. 通过微信/网盘/U盘发送；
3. 对方解压后双击 `run.bat`（会自适应找 python），
   或 `python chapters/ch01_deep_learning_revolution/ch01_polynomial_fitting.py`。

## 方式二：Git 托管（推荐，可协作）
1. 在 Gitee（国内快）或 GitHub 新建一个空仓库；
2. 本地推送：
   ```
   cd C:\Users\eric\Desktop\dlfc-code
   git remote add origin <你的仓库地址>
   git push -u origin master
   ```
3. 对方 `git clone <地址>` 即可。
（注意：git 首次推送需要配置账号密码/令牌，属于你本机的操作。）

## 分享时推荐附带
- `share_gallery.html`：双击用浏览器打开，20 章图集一览；
- 让对方按 README 的"学习建议"顺序阅读；
- 每章脚本自带数值验证 + assert，运行即自检，无需额外验证。

## 注意事项
- 首次运行 Ch10/Ch11 会联网下载 MNIST（约 12MB），可提前把
  `data/mnist/` 目录一起打包避免对方下载。
