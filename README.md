#Mask_Editor
确保安装以下package

pip install PyQt5 SimpleITK numpy scikit-image opencv-python vtk

直接运行程序 myAPP_0516.py

shift+左键-移动

shift+右键-放大

导入要修改边界的mask，为其打勾，选择提取边界，鼠标左键拖拽实现边界移动。完成修改后右键mask文件保存.

衰减半径可控制移动控制点行为的影响范围

右键mask可以生成3d演示模型，右键拖动3d视图实现模型的放大缩小

![演示](https://github.com/Lagranmoon23/Boundary-modification/blob/main/tinywow_gif_80359642.gif)
