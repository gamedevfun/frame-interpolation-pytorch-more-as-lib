import bisect
import os
import torch
import numpy as np
import cv2

from util import load_image


def inference(model_path: str, image_paths: list[str], save_folder: str, inter_frames: int, use_gpu: bool = True, half: bool = True):
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()

    if not half:
        model.float()

    if use_gpu and torch.cuda.is_available():
        if half:
            model = model.half()
        else:
            model.float()
        model = model.cuda()

    os.makedirs(save_folder, exist_ok=True)

    result_frame_index: int = 0

    for i in range(len(image_paths) - 1):
        img1 = image_paths[i]
        img2 = image_paths[i+1]

        img_batch_1, crop_region_1 = load_image(img1)
        img_batch_2, crop_region_2 = load_image(img2)

        img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
        img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)

        results = [
            img_batch_1,
            img_batch_2
        ]

        idxes = [0, inter_frames + 1]
        remains = list(range(1, inter_frames + 1))

        splits = torch.linspace(0, 1, inter_frames + 2)

        for _ in range(len(remains)):
            starts = splits[idxes[:-1]]
            ends = splits[idxes[1:]]
            distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape)
            end_i = start_i + 1

            x0 = results[start_i]
            x1 = results[end_i]

            if use_gpu and torch.cuda.is_available():
                if half:
                    x0 = x0.half()
                    x1 = x1.half()
                x0 = x0.cuda()
                x1 = x1.cuda()

            dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

            with torch.no_grad():
                prediction = model(x0, x1, dt)
            insert_position = bisect.bisect_left(idxes, remains[step])
            idxes.insert(insert_position, remains[step])
            results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
            del remains[step]

        y1, x1, y2, x2 = crop_region_1
        frames = [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]

        last_images_pair: bool = i == len(image_paths) - 2
        for index, frame in enumerate(frames):
            frame_path = os.path.join(save_folder, f"frame_{result_frame_index:09d}.png") 
            if last_images_pair:
                cv2.imwrite(frame_path, frame)
            elif index != len(frames) - 1:
                cv2.imwrite(frame_path, frame)
            result_frame_index += 1
    
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
