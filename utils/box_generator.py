import numpy as np
import torch


def _sample(min_, max_):
    return np.random.randint(min_, max_)

    
class RandomBoxGenerator(object):
    """To generate spatially matched box pairs in the two randomly augmented views.
    """
    def __init__(self, input_size, min_scale, max_scale, num_patches_per_image):
        self.input_size = input_size
        self.min_size = int(min_scale * input_size)
        self.max_size = int(max_scale * input_size)
        self.num_patches_per_image = num_patches_per_image
        # self.center_jitter_size = int(input_size * 0.1) #use for attention guided
        self.center_jitter_size = 10

    def __call__(self):
        box_w = _sample(self.min_size, self.max_size)
        box_h = _sample(self.min_size, self.max_size)

        box_x1 = _sample(0, self.input_size - box_w)
        box_y1 = _sample(0, self.input_size - box_h)

        box_x2 = box_x1 + box_w
        box_y2 = box_y1 + box_h

        # print(box_x1, box_y1, box_x2, box_y2)
        mask = torch.zeros([self.input_size, self.input_size], dtype=int)
        mask[box_y1:box_y2, box_x1:box_x2] = 1

        return mask

    def generate(self,  batch_size):
        # spatial consistency matching requires the transform information of images
        spatial_boxes = []
        
        for i in range(self.num_patches_per_image):
            for batch_idx in range(batch_size):
                # random box generation
                box_w = _sample(self.min_size, self.max_size)
                box_h = _sample(self.min_size, self.max_size)
                box_x1 = _sample(0, self.input_size - box_w)
                box_y1 = _sample(0, self.input_size - box_h)

                box_x2 = box_x1 + box_w
                box_y2 = box_y1 + box_h

                # append a spatial box
                spatial_box = [batch_idx, box_x1, box_y1, box_x2, box_y2]
                spatial_boxes.append(clip_box(spatial_box, self.input_size))

        spatial_boxes = [torch.tensor(spatial_boxes)]

        return spatial_boxes


    def generate_with_center(self,  batch_size, center):
        # spatial consistency matching requires the transform information of images
        spatial_boxes = []
        
        for i in range(self.num_patches_per_image):
            for batch_idx in range(batch_size):
                # random box generation
                box_w = _sample(self.min_size, self.max_size)
                box_h = _sample(self.min_size, self.max_size)
                box_x0 = _sample(center[batch_idx][0] - self.center_jitter_size, center[batch_idx][0] + self.center_jitter_size)
                box_y0 = _sample(center[batch_idx][1]- self.center_jitter_size, center[batch_idx][1] + self.center_jitter_size)
                box_x0 = clip_coordinate(box_x0, self.input_size)
                box_y0 = clip_coordinate(box_y0, self.input_size)

                if box_x0 <= box_w / 2:
                    box_x1, box_x2 = 0, box_w
                elif box_x0 >= self.input_size - box_w / 2:
                    box_x1, box_x2 = self.input_size - box_w, self.input_size - 1
                else:
                    box_x1, box_x2 = box_x0 - box_w / 2, box_x0 + box_w / 2

                if box_y0 <= box_h / 2:
                    box_y1, box_y2 = 0, box_h
                elif box_y0 >= self.input_size - box_h / 2:
                    box_y1, box_y2 = self.input_size - box_h, self.input_size - 1
                else:
                    box_y1, box_y2 = box_y0 - box_h / 2, box_y0 + box_h / 2

                # append a spatial box
                spatial_box = [batch_idx, box_x1, box_y1, box_x2, box_y2]
                spatial_boxes.append(clip_box(spatial_box, self.input_size))

        spatial_boxes = [torch.tensor(spatial_boxes)]

        return spatial_boxes

    def generate_with_center2(self, center):
        # spatial consistency matching requires the transform information of images
        spatial_boxes = []
        
        for batch_idx in range(center.shape[0]):
            # random box generation
            box_w = _sample(self.min_size, self.max_size)
            box_h = _sample(self.min_size, self.max_size)
            box_x0 = center[batch_idx][0]
            box_y0 = center[batch_idx][1]
            
            box_x0 = clip_coordinate(box_x0, self.input_size)
            box_y0 = clip_coordinate(box_y0, self.input_size)

            if box_x0 <= box_w / 2:
                box_x1, box_x2 = 0, box_w
            elif box_x0 >= self.input_size - box_w / 2:
                box_x1, box_x2 = self.input_size - box_w, self.input_size - 1
            else:
                box_x1, box_x2 = box_x0 - box_w / 2, box_x0 + box_w / 2

            if box_y0 <= box_h / 2:
                box_y1, box_y2 = 0, box_h
            elif box_y0 >= self.input_size - box_h / 2:
                box_y1, box_y2 = self.input_size - box_h, self.input_size - 1
            else:
                box_y1, box_y2 = box_y0 - box_h / 2, box_y0 + box_h / 2

            mask = torch.zeros([self.input_size, self.input_size], dtype=int)
            mask[int(box_y1):int(box_y2), int(box_x1):int(box_x2)] = 1

            spatial_boxes.append(mask.unsqueeze(0))

        return spatial_boxes

    def generate_with_center3(self, center):
        # spatial consistency matching requires the transform information of images
        spatial_boxes = []
        
        for batch_idx in range(center.shape[0]):
            # random box generation
            box_w = _sample(self.min_size, self.max_size)
            box_h = _sample(self.min_size, self.max_size)
            box_x0 = center[batch_idx][0]
            box_y0 = center[batch_idx][1]
            
            box_x0 = clip_coordinate(box_x0, self.input_size)
            box_y0 = clip_coordinate(box_y0, self.input_size)

            if box_x0 <= box_w / 2:
                box_x1, box_x2 = 0, box_w
            elif box_x0 >= self.input_size - box_w / 2:
                box_x1, box_x2 = self.input_size - box_w, self.input_size - 1
            else:
                box_x1, box_x2 = box_x0 - box_w / 2, box_x0 + box_w / 2

            if box_y0 <= box_h / 2:
                box_y1, box_y2 = 0, box_h
            elif box_y0 >= self.input_size - box_h / 2:
                box_y1, box_y2 = self.input_size - box_h, self.input_size - 1
            else:
                box_y1, box_y2 = box_y0 - box_h / 2, box_y0 + box_h / 2

            # append a spatial box
            spatial_box = [batch_idx, box_x1, box_y1, box_x2, box_y2]
            spatial_boxes.append(clip_box(spatial_box, self.input_size))

        spatial_boxes = [torch.tensor(spatial_boxes)]

        return spatial_boxes


def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def jitter_box(box_t, box_l, box_b, box_r, box_jittering_ratio, input_size):
    box_w = box_r - box_l
    box_h = box_b - box_t

    jitter = np.random.uniform(low=1. - box_jittering_ratio,
                               high=1. + box_jittering_ratio, size=4)

    box_l = float(box_l + box_w * (jitter[0] - 1))
    box_t = float(box_t + box_h * (jitter[1] - 1))
    box_r = float(box_l + box_w * jitter[2])
    box_b = float(box_t + box_h * jitter[3])

    return box_t, box_l, box_b, box_r


def clip_box(box_with_inds, input_size):
    box_with_inds[1] = float(max(0, box_with_inds[1]))
    box_with_inds[2] = float(max(0, box_with_inds[2]))
    box_with_inds[3] = float(min(input_size, box_with_inds[3]))
    box_with_inds[4] = float(min(input_size, box_with_inds[4]))

    return box_with_inds


def clip_coordinate(coordinate, input_size):
    coordinate = max(0, coordinate)
    coordinate = min(input_size-1, coordinate)

    return coordinate