import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

fwo_thres = 30
max_error_ratio = 0.05
max_error_ratio_2p = 0.5
max_error_ratio_w = 0.05
max_error_ratio_h = 0.05
conf_thres = 0.5
out_thres = 10
fps_out = 100

def get_best_coor(last_coor, list_coor):
    error_list = []
    if len(list_coor) == 1:
        return 0
    for i, coor in enumerate(list_coor):
        error = (int(last_coor[0])-int(coor[0]))**2 + (int(last_coor[1])-int(coor[1]))**2 + (int(last_coor[2])-int(coor[2]))**2 + (int(last_coor[3])-int(coor[3]))**2
        error_list.append(error)
    return np.argmin(error_list)

def get_best_coor_center(last_coor, list_coor):
    error_list = []
    if len(list_coor) == 1:
        return 0
    for i, coor in enumerate(list_coor):
        x_center = (float(coor[0]) + float(coor[2]))//2
        y_center = (float(coor[1]) + float(coor[3]))//2
        error = (last_coor[0]-x_center)**2 + (last_coor[1]-y_center)**2
        error_list.append(error)
    return np.argmin(error_list), np.min(error_list)

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



def cal_distance_and_angle(last, now):
    x_center_now = (float(now[0]) + float(now[2]))//2
    y_center_now = (float(now[1]) + float(now[3]))//2
    x_center_last = (float(last[0]) + float(last[2]))//2
    y_center_last = (float(last[1]) + float(last[3]))//2
    angle = 180*np.arctan2((y_center_now-y_center_last),(x_center_now-x_center_last+1e-8)) / np.pi
    distance = np.sqrt((x_center_now-x_center_last)**2 + (y_center_now-y_center_last)**2)
    return distance, angle, [x_center_now, y_center_now]

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

def detect_image(image, model, names, device, img_size = 512, is_begin = False, last_coor = None, conf_thres = 0.25, iou_thres = 0.6, max_error_ratio = 0.05):
    im0s = image
    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]

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
    t1 = torch_utils.time_synchronized()
    pred = model(img)[0]
    t2 = torch_utils.time_synchronized()

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
                list_conf_0.appand(list_conf[i])
        if len(list_conf_0) > 0 and len(list_conf_1) > 0:
            if is_begin:
                id_best = np.argmax(list_conf)
                coor = list_xyxy[id_best]
                x_center = (float(coor[0]) + float(coor[2]))//2
                y_center = (float(coor[1]) + float(coor[3]))//2
                return (x_center, y_center), list_conf[id_best]
            else:
                id_best = np.argmax(list_conf)
                if list_conf[id_best] > 0.9:
                    coor = list_xyxy[id_best]
                    x_center = (float(coor[0]) + float(coor[2]))//2
                    y_center = (float(coor[1]) + float(coor[3]))//2
                    return (x_center, y_center), list_conf[id_best]
                else:
                    id_best, error = get_best_coor_center(last_coor, list_xyxy)
                    if error < max_error_ratio*img_size:
                        coor = list_xyxy[id_best]
                        x_center = (float(coor[0]) + float(coor[2]))//2
                        y_center = (float(coor[1]) + float(coor[3]))//2
                        return (x_center, y_center), list_conf[id_best]
                    else:
                        return None, None
        else:
            return None, None
    else:
        return None, None

def detect_image_2p(image, model, names, device, img_size = 512, conf_thres = 0.25, iou_thres = 0.6, max_error_ratio = 0.05):
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
                return (x_center, y_center), angle, float(list_conf_1[id1_best])
            else:
                return None, None, None
        else:
            return None, None, None
    else:
        return None, None, None

def detect_video(input_video, out_dir, model, names, colors, device, img_size = 512, conf_thres = 0.25, iou_thres = 0.6):
    out, source = out_dir, input_video
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Set Dataloader
    vid_path, vid_writer = None, None
    save_img = True
    dataset = LoadImages(source, img_size=img_size)

    # Run inference
    t0 = time.time()
    count = 0
    # last_coor = 0
    # last_conf = 0
    # last_cls = 0
    count_fail = 0
    frame_without_obj = 0
    history = []
    continuous_path = ['']
    fps = dataset.fps
    super_conf = False
    for path, img, im0s, vid_cap in dataset:
        p, s, im0 = path, '', im0s
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]
        t2 = torch_utils.time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        #print(pred)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                count+=1
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                #print(len(det))
                list_xyxy = []
                list_conf = []
                list_cls = []
                for *xyxy, conf, cls in det:

                    list_xyxy.append(xyxy)
                    list_conf.append(conf)
                    list_cls.append(cls)
                print(list_cls)
                # print('-------------------------------------------------')
                # print(list(map(int, list_cls))

                if count == 1:
                    id_best = np.argmax(list_conf)
                    coor = list_xyxy[id_best]
                    last_xyxy = coor
                    last_conf = list_conf[id_best]
                    last_cls = list_cls[id_best]
                    frame_without_obj = 0
                else:
                    id_best = np.argmax(list_conf)
                    if list_conf[id_best] > 0.9:
                        coor = list_xyxy[id_best]
                        super_conf = True
                    else:
                        id_best = get_best_coor(last_xyxy, list_xyxy)
                        coor = list_xyxy[id_best]
                    error_x = 0.5*np.abs(float(last_xyxy[0])-float(coor[0])) + 0.5*np.abs(float(last_xyxy[2])-float(coor[2]))
                    error_y = 0.5*np.abs(float(last_xyxy[1])-float(coor[1])) + 0.5*np.abs(float(last_xyxy[3])-float(coor[3]))
                    if (error_x>max_error_ratio_w*im0.shape[1] or error_y>max_error_ratio_h*im0.shape[0]) and frame_without_obj<fwo_thres and super_conf==False:
                        frame_without_obj +=1
                    else:
                        if len(history) > 0:
                            distance, angle, coor_center = cal_distance_and_angle(last_xyxy, coor)
                            velocity = distance*(fps/(frame_without_obj+1))
                        last_xyxy = coor
                        last_conf = list_conf[id_best]
                        last_cls = list_cls[id_best]
                        super_conf = False
                        frame_without_obj = 0
                    
                if frame_without_obj == 0:
                    if count == 1 or len(history) == 0:
                        x_center = (float(last_xyxy[0]) + float(last_xyxy[2]))//2
                        y_center = (float(last_xyxy[1]) + float(last_xyxy[3]))//2
                        coor_center = [x_center, y_center]

                    # content_center += str(coor_center[0]) + ', ' + str(coor_center[1]) + '\n' 
                    history.append(coor_center)
                    continuous_path[-1] = history

                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(last_cls)], last_conf)
                    plot_one_box(last_xyxy, im0, label=label, color=colors[int(last_cls)])

            else:
                frame_without_obj += 1
                count_fail += 1
                if frame_without_obj > out_thres and continuous_path[-1]!='':
                    #print(continuous_path)
                    history = []
                    continuous_path.append('')

            print('%sDone. (%.3fs)  (%.2d)' % (s, t2 - t1, len(continuous_path)))

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                vid_writer.write(im0)

    if save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

def detect_video_multiobj(input_video, out_dir, model, names, colors, device, img_size = 512, conf_thres = 0.25, iou_thres = 0.6):
    out, source = out_dir, input_video
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Set Dataloader
    vid_path, vid_writer = None, None
    save_img = True
    dataset = LoadImages(source, img_size=img_size)

    # Run inference
    t0 = time.time()
    count = 0
    # last_coor = 0
    # last_conf = 0
    # last_cls = 0
    count_fail = 0
    frame_without_obj = 0
    history = []
    continuous_path = ['']
    fps = dataset.fps
    super_conf = False
    for path, img, im0s, vid_cap in dataset:
        p, s, im0 = path, '', im0s
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]
        t2 = torch_utils.time_synchronized()
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        #print(pred)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                count+=1
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                #print(len(det))
                list_xyxy = []
                list_conf = []
                list_cls = []
                for *xyxy, conf, cls in det:

                    list_xyxy.append(xyxy)
                    list_conf.append(conf)
                    list_cls.append(cls)
                # print(list_cls)
                # print('-------------------------------------------------')
                # print(list(map(int, list_cls)))
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
                        list_conf_0.appand(list_conf[i])

                if count == 1:
                    id_best = np.argmax(list_conf)
                    coor = list_xyxy[id_best]
                    last_xyxy = coor
                    last_conf = list_conf[id_best]
                    last_cls = list_cls[id_best]
                    frame_without_obj = 0
                else:
                    id_best = np.argmax(list_conf)
                    if list_conf[id_best] > 0.9:
                        coor = list_xyxy[id_best]
                        super_conf = True
                    else:
                        id_best = get_best_coor(last_xyxy, list_xyxy)
                        coor = list_xyxy[id_best]
                    error_x = 0.5*np.abs(float(last_xyxy[0])-float(coor[0])) + 0.5*np.abs(float(last_xyxy[2])-float(coor[2]))
                    error_y = 0.5*np.abs(float(last_xyxy[1])-float(coor[1])) + 0.5*np.abs(float(last_xyxy[3])-float(coor[3]))
                    if (error_x>max_error_ratio_w*im0.shape[1] or error_y>max_error_ratio_h*im0.shape[0]) and frame_without_obj<fwo_thres and super_conf==False:
                        frame_without_obj +=1
                    else:
                        if len(history) > 0:
                            distance, angle, coor_center = cal_distance_and_angle(last_xyxy, coor)
                            velocity = distance*(fps/(frame_without_obj+1))
                        last_xyxy = coor
                        last_conf = list_conf[id_best]
                        last_cls = list_cls[id_best]
                        super_conf = False
                        frame_without_obj = 0
                    
                if frame_without_obj == 0:
                    if count == 1 or len(history) == 0:
                        x_center = (float(last_xyxy[0]) + float(last_xyxy[2]))//2
                        y_center = (float(last_xyxy[1]) + float(last_xyxy[3]))//2
                        coor_center = [x_center, y_center]

                    # content_center += str(coor_center[0]) + ', ' + str(coor_center[1]) + '\n' 
                    history.append(coor_center)
                    continuous_path[-1] = history

                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(last_cls)], last_conf)
                    plot_one_box(last_xyxy, im0, label=label, color=colors[int(last_cls)])

            else:
                frame_without_obj += 1
                count_fail += 1
                if frame_without_obj > out_thres and continuous_path[-1]!='':
                    #print(continuous_path)
                    history = []
                    continuous_path.append('')

            print('%sDone. (%.3fs)  (%.2d)' % (s, t2 - t1, len(continuous_path)))

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                vid_writer.write(im0)

    if save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

def detect(input_video, out_dir, weights, img_size = 512, cfg = 'cfg/yolov3-tiny3-1cls.cfg', names = 'data/jetbot.names', conf_thres = 0.25, iou_thres = 0.6):
    out, source = out_dir, input_video

    # Initialize
    device = torch_utils.select_device(device='')
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    attempt_download(weights)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Eval mode
    model.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    count = 0
    # last_coor = 0
    # last_conf = 0
    # last_cls = 0
    count_fail = 0
    frame_without_obj = 0
    history = []
    continuous_path = ['']
    error_text = ''
    content_center=''
    content_veloc=''
    content_angle=''
    fps = dataset.fps
    super_conf = False
    for path, img, im0s, vid_cap in dataset:
        p, s, im0 = path, '', im0s
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]
        t2 = torch_utils.time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        #print(pred)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                count+=1
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                #print(len(det))
                list_xyxy = []
                list_conf = []
                list_cls = []
                for *xyxy, conf, cls in det:

                    list_xyxy.append(xyxy)
                    list_conf.append(conf)
                    list_cls.append(cls)

                if count == 1:
                    id_best = np.argmax(list_conf)
                    coor = list_xyxy[id_best]
                    last_xyxy = coor
                    last_conf = list_conf[id_best]
                    last_cls = list_cls[id_best]
                    frame_without_obj = 0
                else:
                    id_best = np.argmax(list_conf)
                    if list_conf[id_best] > 0.9:
                        coor = list_xyxy[id_best]
                        super_conf = True
                    else:
                        id_best = get_best_coor(last_xyxy, list_xyxy)
                        coor = list_xyxy[id_best]
                    error_x = 0.5*np.abs(float(last_xyxy[0])-float(coor[0])) + 0.5*np.abs(float(last_xyxy[2])-float(coor[2]))
                    error_y = 0.5*np.abs(float(last_xyxy[1])-float(coor[1])) + 0.5*np.abs(float(last_xyxy[3])-float(coor[3]))
                    if (error_x>max_error_ratio_w*im0.shape[1] or error_y>max_error_ratio_h*im0.shape[0]) and frame_without_obj<fwo_thres and super_conf==False:
                        frame_without_obj +=1
                    else:
                        if len(history) > 0:
                            distance, angle, coor_center = cal_distance_and_angle(last_xyxy, coor)
                            velocity = distance*(fps/(frame_without_obj+1))
                        last_xyxy = coor
                        last_conf = list_conf[id_best]
                        last_cls = list_cls[id_best]
                        super_conf = False
                        frame_without_obj = 0
                    
                if frame_without_obj == 0:
                    if count == 1 or len(history) == 0:
                        x_center = (float(last_xyxy[0]) + float(last_xyxy[2]))//2
                        y_center = (float(last_xyxy[1]) + float(last_xyxy[3]))//2
                        coor_center = [x_center, y_center]

                    content_center += str(coor_center[0]) + ', ' + str(coor_center[1]) + '\n' 
                    history.append(coor_center)
                    continuous_path[-1] = history

                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(last_cls)], last_conf)
                    plot_one_box(last_xyxy, im0, label=label, color=colors[int(last_cls)])

            else:
                frame_without_obj += 1
                count_fail += 1
                if frame_without_obj > out_thres and continuous_path[-1]!='':
                    #print(continuous_path)
                    history = []
                    continuous_path.append('')

            print('%sDone. (%.3fs)  (%.2d)' % (s, t2 - t1, len(continuous_path)))

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                vid_writer.write(im0)

    if save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


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
        for image_path in image_paths:
            start = time.time()
            image = cv2.imread(image_path)
            a, b, c = detect_image_2p(image, model, names, device)
            print(a, b, c)
            end = time.time()
            print(end-start)
