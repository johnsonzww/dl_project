from PIL import Image

# 加载中间透明的手机图片
base_img = Image.open('D:\Desktop\3.png')
# 新建透明底图，大小和手机图一样，mode使用RGBA，保留Alpha透明度，颜色为透明
# Image.new(mode, size, color=0)，color可以用tuple表示，分别表示RGBA的值
target = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
# box = (166, 64, 320, 337)  # 区域
# 加载需要狐狸像
region = Image.open('D:\Desktop\4.png')
region = region.rotate(180)  # 旋转180度
# 确保图片是RGBA格式，大小和box区域一样
region = region.convert("RGBA")
# region = region.resize((box[2] - box[0], box[3] - box[1]))
# 先将狐狸像合成到底图上
target.paste(region, (0, 0))
# 将手机图覆盖上去，中间透明区域将狐狸像显示出来。
target.paste(base_img, (0, 0), base_img)  # 第一个参数表示需要粘贴的图像，中间的是坐标，最后是一个是mask图片，用于指定透明区域，将底图显示出来。
# target.show()

target.save('./out.png')  # 保存图片
