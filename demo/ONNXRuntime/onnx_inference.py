#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from typing import Dict, Tuple

import cv2
import numpy as np

import json
import pickle
import onnxruntime as ort

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-v",
        "--video_path",
        type=str,
        default=None,
        help="Path to input video file. If not set, use camera.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./thumbs_up_detector.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-cam",
        "--camera_id",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.01,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    return parser

def load_model(model_path) -> Tuple[ort.InferenceSession, Dict[int, str]]:
    """
    Load the ONNX model, first trying as a pickled model, then as a direct ONNX file if that fails.
    Returns:
        Tuple[ort.InferenceSession, Dict[int, str]]: 
            - ONNX Runtime session for inference
            - Class ID to label mapping
    Raises:
        FileNotFoundError: If the model file cannot be found or loaded.
        ValueError: If the model format is invalid.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model_bytes = None
    tried_pickle = False
    # Try loading as pickle first
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        tried_pickle = True
        if isinstance(model_data, bytes):
            model_bytes = model_data
        elif isinstance(model_data, dict):
            model_bytes = model_data.get('model', model_data.get('onnx_model'))
            metadata = session.get_modelmeta()
            print("Model metadata:")
            for key, value in metadata.custom_metadata_map.items():
                print(f"  {key}: {value}")
            id2label = metadata.custom_metadata_map.get("id2label", None)
            if id2label is not None:
                id2label = json.loads(id2label)
            if model_bytes is None:
                raise ValueError("Invalid pickled model format - no model data found")
        else:
            raise ValueError(f"Unsupported pickled model format: {type(model_data)}")
        print("Model loaded as pickled ONNX.")
    except Exception as e:
        print(f"Warning: Could not load as pickled model: {e}")
        print("Attempting to load as direct ONNX file...")
        # Try loading as direct ONNX file
        try:
            with open(model_path, "rb") as f:
                model_bytes = f.read()
            metadata = session.get_modelmeta()
            print("Model metadata:")
            for key, value in metadata.custom_metadata_map.items():
                print(f"  {key}: {value}")
            id2label = metadata.custom_metadata_map.get("id2label", None)
            if id2label is not None:
                id2label = json.loads(id2label)
        except Exception as e2:
            raise FileNotFoundError(f"Failed to load model as pickle ({e}) or as ONNX ({e2})")

    # Set up ONNX Runtime providers (prefer CPU for compatibility)
    providers = [
        "CPUExecutionProvider",
    ]
    # Try to add CUDA if available
    try:
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")
            print("Using CUDA for inference")
        else:
            print("Using CPU for inference")
    except:
        print("Using CPU for inference")

    # Create inference session
    session = ort.InferenceSession(model_bytes, providers=providers)

    metadata = session.get_modelmeta()
    old = metadata.custom_metadata_map.get("id2label", None)
    id2label = {}
    if old is not None:
        old = json.loads(old)
        for i, item in old.items():
            id2label[int(i)] = item

    print(f"Model loaded successfully with {len(id2label)} classes: {id2label}")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    print(f"Input name: {session.get_inputs()[0].name}")

    return session, id2label

if __name__ == '__main__':
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    session, id2label = load_model(args.model)

    if args.video_path:
        # Video file inference
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {args.video_path}")
            exit(1)
        # Prepare output video writer
        os.makedirs(args.output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.video_path))[0] + '_out.mp4')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Processing video. Output will be saved to {out_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            origin_img = frame.copy()
            img, ratio = preprocess(origin_img, input_shape)
            ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
            output = session.run(None, ort_inputs)
            predictions = demo_postprocess(output[0], input_shape)[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]
            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
            boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=args.score_thr)
            vis_img = origin_img.copy()
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                vis_img = vis(vis_img, final_boxes, final_scores, final_cls_inds,
                              conf=args.score_thr, class_names=id2label)
            writer.write(vis_img)
            idx += 1
            if idx % 10 == 0:
                print(f"Processed {idx}/{frame_count} frames...")
        cap.release()
        writer.release()
        print(f"Video saved to {out_path}")
    else:
        # Camera inference
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera_id}")
            exit(1)

        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break

            origin_img = frame.copy()
            img, ratio = preprocess(origin_img, input_shape)
            ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
            output = session.run(None, ort_inputs)
            predictions = demo_postprocess(output[0], input_shape)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
            boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=args.score_thr)
            vis_img = origin_img.copy()
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                vis_img = vis(vis_img, final_boxes, final_scores, final_cls_inds,
                              conf=args.score_thr, class_names=id2label)

            cv2.imshow('Camera', vis_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
