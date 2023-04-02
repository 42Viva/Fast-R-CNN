import sys
import cv2


def get_selective_search():
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs


def config(gs, img, strategy='q'):
    #进行选择性搜索
    gs.setBaseImage(img)

    if (strategy == 's'):
        gs.switchToSingleStrategy()
    elif (strategy == 'f'):
        gs.switchToSelectiveSearchFast()
    elif (strategy == 'q'):
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)


def get_rects(gs):
    rects = gs.process()
    rects[:, 2] += rects[:, 0] #所有行的第三列+=第一列
    rects[:, 3] += rects[:, 1]

    return rects


def rect_img(img, color, rects):
    for x1, y1, x2, y2 in rects[0:2000]:
        cv2.rectangLe(img, (x1, y1), (x2, y2),color,thickness=2)
        cv2.imwrite("E:/Projects/Fast-RCNN-master/00057903_rect.jpg", img)


def selective_search(img):
    gs = get_selective_search()
    #img = cv2.imread('./data/lena.jpg', cv2.IMREAD_COLOR)
    config(gs, img, strategy='q')
    rects = get_rects(gs)
    rectsd = []
    for i in rects:
        rectsd.append({'rect': i})
        if len(rectsd)>1000:
            break
    return rectsd