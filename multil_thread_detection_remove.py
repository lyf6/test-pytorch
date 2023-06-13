"""Add slate information dynamically.

- Open template.
- Update info.
- Save as jpg.
- Close current document.

"""

# Import built-in modules
from photoshop import Session
import photoshop.api as ps
from datetime import datetime
import os
from tempfile import mkdtemp
import numpy as np
import cv2
from PIL import Image
# Import third-party modules
# import examples._psd_files as psd  # Import from examples.
import onnxruntime
import torch
import time
import keyboard

import shutil
UAVCAR_CLASSES = (
    "car"
)
_COLORS = np.array(
    [
        0.000, 0.447, 0.741
    ]
).astype(np.float32).reshape(-1, 3)
# Import local modules


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def save_local(save_dir='D:\code\photoshop-python-api//tmp'):
    with Session() as ps:
        doc = ps.active_document
        mkdir(save_dir)
        image_path = os.path.join(save_dir, f"{doc.name}")
        doc.saveAs(image_path, ps.TiffSaveOptions())


def load2ps(image_path):
    with Session() as ps:
        desc = ps.ActionDescriptor
        desc.putPath(ps.app.charIDToTypeID("null"),
                     image_path)
        ps.app.executeAction(ps.app.charIDToTypeID("Plc "), desc)


def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones(
            (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None],
                valid_cls_inds[keep, None]], 1
        )
    return dets


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(
            _COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]),
                    font, 0.4, txt_color, thickness=1)

    return img


def detection(session, image_path, input_shape=(640, 640)):

    origin_img = cv2.imread(image_path)
    img, ratio = preprocess(origin_img, input_shape)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.4)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:,
                                                         :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=0.4, class_names='car')
        cv2.imwrite('detection.jpg', origin_img)
        return final_boxes
    else:
        return []


def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


def resize_max_size(
    np_img, size_limit: int, interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    # Resize image's longer size to size_limit if longer size larger than size_limit
    h, w = np_img.shape[:2]
    if max(h, w) > size_limit:
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)
        new_h = int(h * ratio + 0.5)
        return cv2.resize(np_img, dsize=(new_w, new_h), interpolation=interpolation)
    else:
        return np_img


def erase_process(image_path, dets):

    image = cv2.imread(image_path)
    origin_image = image
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    #np.squeeze(np.array(Image.open(image_path), dtype=np.uint8))

    original_shape = image.shape
    height, width, _ = image.shape
    mask = np.zeros(shape=(height, width), dtype=np.uint8)
    origin_mask = np.zeros(shape=(height, width), dtype=np.uint8)
    max_scale = 10
    min_scale = 5
    for obj in dets:
        origin_xmin = max(int(obj[0] - min_scale), 0)
        origin_ymin = max(int(obj[1] - min_scale), 0)
        origin_xmax = min(int(obj[2] + min_scale), width)
        origin_ymax = min(int(obj[3] + min_scale), height)
        xmin = max(int(obj[0] - max_scale), 0)
        ymin = max(int(obj[1] - max_scale), 0)
        xmax = min(int(obj[2] + max_scale), width)
        ymax = min(int(obj[3] + max_scale), height)
        mask[ymin:ymax, xmin: xmax] = 255
        origin_mask[origin_ymin:origin_ymax, origin_xmin:origin_xmax] = 255
    # origin_mask = mask
    interpolation = cv2.INTER_CUBIC

    size_limit = 2000
    if size_limit == "Original":
        size_limit = max(image.shape)
    else:
        size_limit = int(size_limit)

    print(f"Origin image shape: {original_shape}")
    image = resize_max_size(image, size_limit=size_limit,
                            interpolation=interpolation)
    print(f"Resized image shape: {image.shape}")
    image = norm_img(image)
    mask = resize_max_size(mask, size_limit=size_limit,
                           interpolation=interpolation)
    mask = norm_img(mask)
    return image, mask, origin_image, origin_mask


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def erase_run(model, image, mask, device="cuda:0"):
    """
    image: [C, H, W]
    mask: [1, H, W]
    return: BGR IMAGE
    """
    origin_height, origin_width = image.shape[1:]
    image = pad_img_to_modulo(image, mod=8)
    mask = pad_img_to_modulo(mask, mod=8)

    mask = (mask > 0) * 1
    # mask.astype(np.uint8)
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        torch.cuda.empty_cache()
        inpainted_image = model(image, mask)
        torch.cuda.empty_cache()
    print(f"process time: {(time.time() - start)*1000}ms")
    cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = cur_res[0:origin_height, 0:origin_width, :]
    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_BGR2RGB)
    return cur_res


def erase(detection_model, erase_model, image_path, device="cuda:0"):
    dets = detection(detection_model, image_path)
    if len(dets) > 0:
        image, mask, origin_image, origin_mask = erase_process(
            image_path, dets)
        erase_img = erase_run(erase_model, image, mask, device)
        new_height, new_width, _ = erase_img.shape
        ori_height, ori_widrh, _ = origin_image.shape
        if new_height == ori_height and new_width == ori_widrh:
            # cv2.imwrite(save_path, erase_img)
            # cv2.imwrite('mask.jpg', origin_mask)
            origin_mask = origin_mask[..., None] > 0
            # img1 = erase_img * origin_mask
            # img2 = origin_image * (1 - origin_mask)
            # cv2.imwrite('erase_img_mask.jpg', img1)
            # cv2.imwrite('origin_image.jpg', img2)

            img = erase_img * origin_mask + origin_image * (1 - origin_mask)
            return img
        else:
            interpolation = cv2.INTER_CUBIC
            erase_img = cv2.resize(erase_img, dsize=(ori_widrh, ori_height),
                                   interpolation=interpolation)

            #erase_img = erase_img.transpose((1, 0, 2))
            # cv2.imwrite('mask.jpg', origin_mask)
            origin_mask = origin_mask[..., None] > 0
            # img1 = erase_img * origin_mask
            # img2 = origin_image * (1 - origin_mask)
            # cv2.imwrite('erase_img_mask.jpg', img1)
            # cv2.imwrite('origin_image.jpg', img2)

            img = erase_img * origin_mask + origin_image * (1 - origin_mask)
            return img
    else:
        img = cv2.imread(image_path)
        return img


def start(detection_model, erase_model, image_path, save_dir):

    img = erase(detection_model, erase_model, image_path)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = img.transpose((2, 1, 0))
    img = Image.fromarray(img)
    # img_exif = Image.open(image_path).info['exif']
    new_file = os.path.join(save_dir, os.path.basename(image_path))

    # img.save(new_file, "jpeg", exif=img_exif)
    img.save(new_file)


if __name__ == '__main__':
    detection_model = './models/yolox.onnx'
    erase_model = './models/big-lama.pt'
    device = "cuda:0"
    detection_model = onnxruntime.InferenceSession(detection_model)
    erase_model = torch.jit.load(erase_model, map_location="cpu")
    erase_model = erase_model.to(device)
    erase_model.eval()
    # dirp = 'MP'
    img_dir = 'H://test_data\image'
    save_dir = 'H://test_data//result//'
    # mkdir(save_dir)
    img_list = os.listdir(img_dir)
    for img in img_list:
        img_path = os.path.join(img_dir, img)
        start(detection_model, erase_model, img_path, save_dir)
