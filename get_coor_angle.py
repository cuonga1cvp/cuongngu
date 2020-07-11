import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

max_error_ratio_2p = 0.5

def get_best_coor_with2p(list_coor_0, list_coor_1):
    for i, coor_1 in enumerate(list_coor_1):
        # ps_dig = np.sqrt((float(coor_1[2]) - float(coor_1[0]))**2 + (float(coor_1[2]) - float(coor_1[0]))**2)
        x_center_1 = (float(coor_1[0]) + float(coor_1[2]))//2
        y_center_1 = (float(coor_1[1]) + float(coor_1[3]))//2
        for j, coor_0 in enumerate(list_coor_0):
            x_center_0 = (float(coor_0[0]) + float(coor_0[2]))//2
            y_center_0 = (float(coor_0[1]) + float(coor_0[3]))//2
            error = (x_center_1-x_center_0)**2 + (y_center_1-y_center_0)**2
            #error_list.append(error)
            if i == j == 0:
                error_min = error
                id1_min = i
                id0_min = j
            elif error < error_min:
                error_min = error
                id1_min = i
                id0_min = j
    return id0_min, id1_min, error_min

def get_real_distance(lip, dis2obj, sen_size=(8.5,8.5), sen_pixel=(1920,1080), focal_len=3.34):
    wos = sen_size[0]*np.abs(lip[1][0] - lip[0][0])/sen_pixel[0]
    hos = sen_size[1]*np.abs(lip[1][1] - lip[0][1])/sen_pixel[1]

    rw = dis2obj*wos/focal_len
    hw = dis2obj*hos/focal_len

    return np.sqrt(rw**2 + hw**2)

def init_model(weights, img_size = 512, cfg = 'cfg/yolov3-tiny3-1cls.cfg', names = 'data/jetbot.names'):
    # Initialize
    device = torch_utils.select_device(device='')

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    attempt_download(weights)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Eval mode
    model.to(device).eval()

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    return model, names, colors, device

def detect_image_2p(image, model, names, device, img_size = 512, conf_thres = 0.25, iou_thres = 0.6, max_error_ratio_2p = 0.05, is_save = False, line_thickness=8, path_save='img_out.jpg'):
    im0s = image
    # Padded resize
    img = letterbox(im0s, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    s, im0 = '', im0s
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
    det = pred[0]
    if det is not None and len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
        
        list_xyxy = []
        list_conf = []
        list_cls = []
        for *xyxy, conf, cls in det:

            list_xyxy.append(xyxy)
            list_conf.append(conf)
            list_cls.append(cls)

        list_cls_int = list(map(int, list_cls))
        list_xyxy_0 = []
        list_xyxy_1 = []
        list_conf_0 = []
        list_conf_1 = []
        for i, cls_int in enumerate(list_cls_int):
            if cls_int == 1:
                list_xyxy_1.append(list_xyxy[i])
                list_conf_1.append(list_conf[i])
            else:
                list_xyxy_0.append(list_xyxy[i])
                list_conf_0.append(list_conf[i])
        if len(list_conf_0) > 0 and len(list_conf_1) > 0:
            id0_best, id1_best, error = get_best_coor_with2p(list_xyxy_0, list_xyxy_1)
            coor_1 = list_xyxy_1[id1_best]
            coor_0 = list_xyxy_0[id0_best]
            diag = np.sqrt((float(coor_1[2]) - float(coor_1[0]))**2 + (float(coor_1[2]) - float(coor_1[0]))**2)
            if error < max_error_ratio_2p*diag:
                x_center = (float(coor_1[0]) + float(coor_1[2]))//2
                y_center = (float(coor_1[1]) + float(coor_1[3]))//2
                x_center_0 = (float(coor_0[0]) + float(coor_0[2]))//2
                y_center_0 = (float(coor_0[1]) + float(coor_0[3]))//2
                angle = (np.arctan2(x_center_0 - x_center, y_center_0-y_center)/np.pi) * 180
                if is_save:
                    cv2.line(im0s, (int(x_center), int(y_center)), (int(x_center_0), int(y_center_0)), (0, 255, 0), thickness=line_thickness)
                    cv2.imwrite(path_save, im0s)
                return (x_center, y_center), angle, float(list_conf_1[id1_best])
            else:
                return None, None, None
        else:
            return None, None, None
    else:
        return None, None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny3-1cls.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/jetbot.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/0505_last_yolov3-tiny3-1cls.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    #parser.add_argument('--source', type=str, default='/home/thanhtt/Downloads/jetBot', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='/home/thanhtt/Downloads/yolov3_output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    
    with torch.no_grad():
        model, names, colors, device = init_model(opt.weights, cfg = opt.cfg, names=opt.names)
        # detect_video(opt.source, opt.output, model, names, colors, device)
        image_names = next(os.walk(opt.source))[2]
        image_paths = [os.path.join(opt.source, aa) for aa in image_names]
        if os.path.exists(opt.source+'_out'):
            shutil.rmtree(opt.source+'_out')
        os.mkdir(opt.source+'_out')
        image_out = [os.path.join(opt.source+'_out', aa) for aa in image_names]
        for i, image_path in enumerate(image_paths):
            start = time.time()
            image = cv2.imread(image_path)
            a, b, c = detect_image_2p(image, model, names, device, is_save=True, path_save=image_out[i], max_error_ratio_2p=4.0)
            print(a, b, c)
            end = time.time()
            print(end-start)